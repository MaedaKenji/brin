"""
gps2.py — Dynamic POI Direction Awareness System.

Location-agnostic version of gps.py. Works with ANY POI CSV file, not just
the NCU-specific one. Supports per-CSV caching for multi-location scenarios.

Identifies buildings / POIs around any given coordinate and determines their
relative compass direction.

Usage as importable module:
    from gps2 import query, load_pois

    # Default CSV (backward compatible — uses poi_buildings_english.csv)
    result = query(24.968, 121.191, radius_m=500)

    # Custom CSV (e.g., from fetch_poi2.py)
    result = query(35.6812, 139.7671, radius_m=500, csv_path="tokyo_pois.csv")

Usage as standalone FastAPI server:
    uvicorn gps2:app --reload --port 8000
    # Then POST to http://localhost:8000/direction-poi
"""

import csv
import math
import os
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

# ── Constants ────────────────────────────────────────────────────────────────────

EARTH_RADIUS_M = 6_371_000  # Earth radius in meters

# Default POI CSV (backward compatible with original gps.py)
DEFAULT_POI_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "poi_buildings_english.csv"
)

# 8-way compass direction sectors (bearing ranges in degrees)
# North wraps around 0°, so it's handled specially
DIRECTION_SECTORS = [
    ("North",      337.5, 360.0),
    ("North",        0.0,  22.5),
    ("North-East",  22.5,  67.5),
    ("East",        67.5, 112.5),
    ("South-East", 112.5, 157.5),
    ("South",      157.5, 202.5),
    ("South-West", 202.5, 247.5),
    ("West",       247.5, 292.5),
    ("North-West", 292.5, 337.5),
]

ALL_DIRECTIONS = [
    "North", "North-East", "East", "South-East",
    "South", "South-West", "West", "North-West",
]


# ── POI Data ─────────────────────────────────────────────────────────────────────

def load_pois(csv_path: Optional[str] = None) -> list[dict]:
    """
    Load POI data from CSV file.

    Args:
        csv_path: Path to the POI CSV file. If None, uses the default
                  poi_buildings_english.csv.

    Returns:
        List of dicts with keys: id, name, lat, lng
        Optional additional keys (category, description, floors) if present.
    """
    path = csv_path or DEFAULT_POI_CSV

    if not os.path.exists(path):
        return []

    pois = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for row in reader:
            poi = {
                "id":   int(row.get("id", len(pois) + 1)),
                "name": row["name"],
                "lat":  float(row["lat"]),
                "lng":  float(row["lng"]),
            }
            # Include optional fields if they exist
            if "category" in fieldnames and row.get("category"):
                poi["category"] = row["category"]
            if "description" in fieldnames and row.get("description"):
                poi["description"] = row["description"]
            if "floors" in fieldnames and row.get("floors"):
                poi["floors"] = row["floors"]

            pois.append(poi)

    return pois


def build_kdtree(pois: list[dict]) -> tuple[KDTree, np.ndarray]:
    """
    Build a KDTree from POI coordinates for fast spatial queries.

    Converts lat/lng to 3D Cartesian (ECEF) coordinates for accurate
    distance-based nearest-neighbor queries on a sphere.

    Returns:
        (kdtree, coords_rad) where coords_rad is Nx2 array of [lat_rad, lng_rad]
    """
    coords_rad = np.array([
        [math.radians(p["lat"]), math.radians(p["lng"])] for p in pois
    ])

    # Convert to 3D Cartesian for KDTree (more accurate than raw lat/lng)
    cart = np.array([
        [
            math.cos(lat) * math.cos(lng),
            math.cos(lat) * math.sin(lng),
            math.sin(lat),
        ]
        for lat, lng in coords_rad
    ])

    tree = KDTree(cart)
    return tree, coords_rad


# ── Spatial math ─────────────────────────────────────────────────────────────────

def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate great-circle distance between two GPS points using Haversine formula.

    Args:
        lat1, lng1: First point in decimal degrees
        lat2, lng2: Second point in decimal degrees

    Returns:
        Distance in meters
    """
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = math.sin(dlat / 2) ** 2 + \
        math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_M * c


def bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate initial bearing (azimuth) from point 1 to point 2.

    Args:
        lat1, lng1: Origin point in decimal degrees
        lat2, lng2: Destination point in decimal degrees

    Returns:
        Bearing in degrees [0, 360) where 0=North, 90=East, 180=South, 270=West
    """
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

    dlng = lng2 - lng1

    x = math.sin(dlng) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(dlng)

    angle = math.degrees(math.atan2(x, y))
    return angle % 360  # Normalize to [0, 360)


def bearing_to_direction(bearing_deg: float) -> str:
    """
    Convert a bearing angle to an 8-way compass direction label.

    Args:
        bearing_deg: Bearing in degrees [0, 360)

    Returns:
        Direction string, e.g. "North", "South-East", etc.
    """
    for direction, low, high in DIRECTION_SECTORS:
        if low <= bearing_deg < high:
            return direction
    return "North"  # Fallback (should not happen)


# ── Core query ───────────────────────────────────────────────────────────────────

def get_nearby_pois(
    lat: float,
    lng: float,
    radius_m: float = 500,
    pois: Optional[list[dict]] = None,
    tree: Optional[KDTree] = None,
    csv_path: Optional[str] = None,
) -> dict:
    """
    Find POIs near a GPS coordinate, grouped by compass direction.

    Args:
        lat:      Input latitude (decimal degrees)
        lng:      Input longitude (decimal degrees)
        radius_m: Search radius in meters (default 500)
        pois:     Pre-loaded POI list (if None, loads from CSV)
        tree:     Pre-built KDTree (if None, builds one)
        csv_path: Path to POI CSV file (if pois is None)

    Returns:
        Dictionary with structure:
        {
            "input": {"lat": ..., "lng": ...},
            "radius_m": ...,
            "directions": {
                "North": [{"name": ..., "distance_m": ..., "bearing": ..., "lat": ..., "lng": ...}],
                ...
            },
            "nearest": {"name": ..., "distance_m": ..., "direction": ...} or None,
            "total_pois_found": int
        }
    """
    if pois is None:
        pois = load_pois(csv_path)

    if not pois:
        return _empty_result(lat, lng, radius_m)

    # Build KDTree if not provided
    if tree is None:
        tree, _ = build_kdtree(pois)

    # Convert input to 3D Cartesian for KDTree query
    lat_rad = math.radians(lat)
    lng_rad = math.radians(lng)
    query_point = [
        math.cos(lat_rad) * math.cos(lng_rad),
        math.cos(lat_rad) * math.sin(lng_rad),
        math.sin(lat_rad),
    ]

    # Convert radius to Euclidean chord distance for KDTree
    # chord = 2 * sin(angle/2), angle = radius / R
    angle = radius_m / EARTH_RADIUS_M
    chord = 2 * math.sin(angle / 2)

    # Query KDTree for all points within radius
    indices = tree.query_ball_point(query_point, chord)

    if not indices:
        return _empty_result(lat, lng, radius_m)

    # Build direction-grouped results
    directions = {d: [] for d in ALL_DIRECTIONS}
    all_found = []

    for idx in indices:
        poi = pois[idx]
        dist = haversine(lat, lng, poi["lat"], poi["lng"])

        # Double-check with accurate Haversine (KDTree uses chord approx)
        if dist > radius_m:
            continue

        brng = bearing(lat, lng, poi["lat"], poi["lng"])
        direction = bearing_to_direction(brng)

        entry = {
            "name":       poi["name"],
            "distance_m": round(dist, 1),
            "bearing":    round(brng, 1),
            "lat":        poi["lat"],
            "lng":        poi["lng"],
        }

        directions[direction].append(entry)
        all_found.append(entry)

    # Sort each direction by distance (nearest first)
    for d in directions:
        directions[d].sort(key=lambda x: x["distance_m"])

    # Find overall nearest
    nearest = None
    if all_found:
        nearest_entry = min(all_found, key=lambda x: x["distance_m"])
        nearest = {
            "name":       nearest_entry["name"],
            "distance_m": nearest_entry["distance_m"],
            "direction":  bearing_to_direction(nearest_entry["bearing"]),
        }

    return {
        "input":           {"lat": lat, "lng": lng},
        "radius_m":        radius_m,
        "directions":      directions,
        "nearest":         nearest,
        "total_pois_found": len(all_found),
    }


def _empty_result(lat: float, lng: float, radius_m: float) -> dict:
    """Return an empty result structure."""
    return {
        "input":           {"lat": lat, "lng": lng},
        "radius_m":        radius_m,
        "directions":      {d: [] for d in ALL_DIRECTIONS},
        "nearest":         None,
        "total_pois_found": 0,
    }


# ── Per-CSV-path singleton cache for fast repeated queries ───────────────────────

# Cache keyed by absolute CSV path so multiple location datasets can coexist
_cache: dict[str, dict] = {}


def _ensure_loaded(csv_path: Optional[str] = None) -> tuple[list[dict], KDTree]:
    """
    Load POI data and build KDTree if not already cached for the given path.

    Returns:
        (pois, tree)
    """
    resolved = os.path.abspath(csv_path or DEFAULT_POI_CSV)

    if resolved not in _cache:
        pois = load_pois(resolved)
        if pois:
            tree, _ = build_kdtree(pois)
        else:
            tree = None
        _cache[resolved] = {"pois": pois, "tree": tree}

    entry = _cache[resolved]
    return entry["pois"], entry["tree"]


def invalidate_cache(csv_path: Optional[str] = None) -> None:
    """
    Invalidate the cached data for a given CSV path, or all caches if None.

    Call this after updating a POI CSV to force reload on next query.
    """
    if csv_path is None:
        _cache.clear()
    else:
        resolved = os.path.abspath(csv_path)
        _cache.pop(resolved, None)


def query(
    lat: float,
    lng: float,
    radius_m: float = 500,
    csv_path: Optional[str] = None,
) -> dict:
    """
    Convenience function: query nearby POIs with cached data.

    This is the simplest way to use this module:
        from gps2 import query

        # Default (backward compatible with gps.py)
        result = query(24.968, 121.191)

        # Custom CSV
        result = query(35.6812, 139.7671, csv_path="tokyo_pois.csv")
    """
    pois, tree = _ensure_loaded(csv_path)
    return get_nearby_pois(lat, lng, radius_m, pois=pois, tree=tree)


# ── FastAPI Application ──────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="Dynamic POI Direction API",
        description="Find buildings and POIs grouped by compass direction for any location.",
        version="2.0.0",
    )

    class DirectionQuery(BaseModel):
        lat: float = Field(..., description="Latitude in decimal degrees", examples=[24.968])
        lng: float = Field(..., description="Longitude in decimal degrees", examples=[121.191])
        radius: float = Field(500, description="Search radius in meters", ge=1, le=5000)
        csv_path: Optional[str] = Field(
            None,
            description="Path to POI CSV file. If not provided, uses the default.",
        )

    @app.on_event("startup")
    def startup():
        _ensure_loaded()

    @app.get("/health")
    def health():
        pois, _ = _ensure_loaded()
        return {"status": "ok", "pois_loaded": len(pois)}

    @app.get("/pois")
    def list_pois(csv_path: Optional[str] = None):
        pois, _ = _ensure_loaded(csv_path)
        return {"total": len(pois), "pois": pois}

    @app.post("/direction-poi")
    def direction_poi(q: DirectionQuery):
        return query(q.lat, q.lng, q.radius, csv_path=q.csv_path)

except ImportError:
    # FastAPI not installed — module still works as importable library
    app = None


# ── CLI entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    import io

    # Fix Windows console encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("Dynamic POI Direction Awareness System (v2)")
    print("=" * 60)

    # Load default data
    pois, _ = _ensure_loaded()
    print(f"Loaded {len(pois)} POIs from default CSV")

    # Test cases
    test_cases = [
        ("Center of NCU campus",  24.968,  121.191,  500, None),
        ("Near NCU Library",      24.9683, 121.1943, 200, None),
        ("Outside campus",        0.0,     0.0,      500, None),
    ]

    # Also test with poi_seed.csv if it exists
    seed_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poi_seed.csv")
    if os.path.exists(seed_csv):
        pois_seed, _ = _ensure_loaded(seed_csv)
        print(f"Also loaded {len(pois_seed)} POIs from poi_seed.csv")
        test_cases.append(("poi_seed.csv test", 24.968, 121.191, 500, seed_csv))

    for label, lat, lng, radius, csv_path in test_cases:
        print(f"\n{'─' * 60}")
        print(f"Test: {label}")
        print(f"Input: lat={lat}, lng={lng}, radius={radius}m")
        if csv_path:
            print(f"CSV: {csv_path}")

        result = query(lat, lng, radius, csv_path=csv_path)

        print(f"Total POIs found: {result['total_pois_found']}")
        if result["nearest"]:
            n = result["nearest"]
            print(f"Nearest: {n['name']} ({n['distance_m']}m, {n['direction']})")

        for direction in ALL_DIRECTIONS:
            entries = result["directions"][direction]
            if entries:
                names = [f"{e['name']} ({e['distance_m']}m)" for e in entries[:3]]
                suffix = f" +{len(entries)-3} more" if len(entries) > 3 else ""
                print(f"  {direction:12s}: {', '.join(names)}{suffix}")

    # Print full JSON for center of campus
    print(f"\n{'═' * 60}")
    print("Full JSON output (center of campus):")
    print(json.dumps(query(24.968, 121.191, 300), indent=2, ensure_ascii=False))
