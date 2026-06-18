"""
fetch_poi2.py — Dynamic & Flexible POI Fetcher for Any Location.

Queries the Overpass API for buildings/POIs within a dynamically-computed
bounding box around any given coordinate. Supports two modes:

  1. Manual:  Provide --lat / --lng / --radius directly.
  2. Auto:    Use --auto to detect current location via IP geolocation.

This is the "Fase 1: Geolocation Seeding" step from the PRD.

Usage:
    # Auto-detect location
    python fetch_poi2.py --auto

    # Manual — NCU campus
    python fetch_poi2.py --lat 24.968 --lng 121.191 --radius 1000

    # Manual — Tokyo Station, small radius
    python fetch_poi2.py --lat 35.6812 --lng 139.7671 --radius 500 --output tokyo_pois.csv

    # As importable module
    from fetch_poi2 import fetch_pois, get_current_location
    pois = fetch_pois(lat=24.968, lng=121.191, radius_m=1000)
"""

import argparse
import csv
import io
import math
import re
import sys
import time
from typing import Optional

import requests
from unidecode import unidecode

# Fix Windows console encoding for non-ASCII characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Constants ────────────────────────────────────────────────────────────────────

DEFAULT_RADIUS_M = 1000  # 1 km default search radius
DEFAULT_OUTPUT = "poi_seed.csv"

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
IP_GEOLOCATION_URL = "http://ip-api.com/json/?fields=status,message,lat,lon,city,regionName,country"

OVERPASS_HEADERS = {
    "User-Agent": "DynamicPOIFetcher/2.0",
    "Accept": "application/json",
}

# ── IP-based auto-location ───────────────────────────────────────────────────────


def get_current_location() -> dict:
    """
    Detect the current location using IP-based geolocation.

    Returns:
        dict with keys: lat, lng, city, region, country

    Raises:
        RuntimeError: If geolocation service fails or returns an error.
    """
    try:
        resp = requests.get(IP_GEOLOCATION_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach geolocation service: {e}")

    if data.get("status") != "success":
        msg = data.get("message", "Unknown error")
        raise RuntimeError(f"Geolocation service error: {msg}")

    return {
        "lat": data["lat"],
        "lng": data["lon"],
        "city": data.get("city", ""),
        "region": data.get("regionName", ""),
        "country": data.get("country", ""),
    }


# ── Dynamic bounding box ────────────────────────────────────────────────────────


def compute_bounding_box(
    lat: float, lng: float, radius_m: float
) -> tuple[float, float, float, float]:
    """
    Compute a bounding box (min_lat, min_lng, max_lat, max_lng) from a center
    coordinate and a radius in metres.

    Uses a simple spherical approximation (accurate enough for radii < 50 km).
    """
    # Earth radius in metres
    R = 6_371_000

    # Angular distance in radians
    d_lat = radius_m / R
    d_lng = radius_m / (R * math.cos(math.radians(lat)))

    min_lat = lat - math.degrees(d_lat)
    max_lat = lat + math.degrees(d_lat)
    min_lng = lng - math.degrees(d_lng)
    max_lng = lng + math.degrees(d_lng)

    return (min_lat, min_lng, max_lat, max_lng)


# ── Overpass query builder ───────────────────────────────────────────────────────


def build_overpass_query(bbox: tuple[float, float, float, float]) -> str:
    """
    Build an Overpass API query that fetches POI-relevant features inside
    the given bounding box.

    Fetches: buildings, amenities, shops, tourism, leisure, offices, and
    named nodes.

    Args:
        bbox: (min_lat, min_lng, max_lat, max_lng)

    Returns:
        Overpass QL query string.
    """
    min_lat, min_lng, max_lat, max_lng = bbox
    bb = f"{min_lat},{min_lng},{max_lat},{max_lng}"

    return f"""
[out:json][timeout:60];
(
  way["building"]({bb});
  relation["building"]({bb});
  node["building"]({bb});
  node["amenity"]({bb});
  way["amenity"]({bb});
  node["shop"]({bb});
  way["shop"]({bb});
  node["tourism"]({bb});
  way["tourism"]({bb});
  node["leisure"]({bb});
  way["leisure"]({bb});
  node["office"]({bb});
  way["office"]({bb});
);
out center;
"""


# ── POI extraction & deduplication ───────────────────────────────────────────────

# Regex pattern for generic fallback names like "Building (123456)" or "Industrial (789)"
_GENERIC_NAME_PATTERN = re.compile(
    r"^(Building|Industrial|Apartments|House|Shed|Warehouse)\s*\(\d+\)$", re.IGNORECASE
)


def _transliterate_name(name: str) -> str:
    """
    Transliterate non-ASCII characters (Chinese, Japanese, etc.) to their
    closest ASCII equivalents using unidecode.

    Preserves already-ASCII text. Cleans up extra whitespace.
    """
    if not name:
        return name

    # Check if any non-ASCII characters exist
    if all(ord(c) < 128 for c in name):
        return name  # Already English, no change needed

    transliterated = unidecode(name)
    # Clean up extra whitespace from transliteration
    transliterated = re.sub(r"\s+", " ", transliterated).strip()
    return transliterated


def _is_generic_unnamed(name: str) -> bool:
    """
    Check if a name is a generic auto-generated fallback (e.g. 'Building (12345)').

    These carry no useful information for downstream tasks like review enrichment.
    """
    return bool(_GENERIC_NAME_PATTERN.match(name))


def _extract_name(tags: dict) -> Optional[str]:
    """Try to extract a human-readable name from OSM tags."""
    # Prefer English name, then generic name, then official/alt names
    name = (
        tags.get("name:en")
        or tags.get("name")
        or tags.get("official_name")
        or tags.get("alt_name")
    )
    if name:
        return name.strip()

    # Fall back to amenity/shop/tourism/leisure type
    for key in ("amenity", "shop", "tourism", "leisure", "office"):
        value = tags.get(key)
        if value and value != "yes":
            return value.replace("_", " ").title()

    return None


def _extract_category(tags: dict) -> str:
    """Determine a POI category from OSM tags."""
    if tags.get("amenity"):
        amenity = tags["amenity"]
        if amenity in ("restaurant", "cafe", "fast_food", "food_court", "bar", "pub"):
            return "food_and_dining"
        if amenity in ("atm", "bank"):
            return "banking"
        if amenity in ("university", "school", "college", "library", "kindergarten"):
            return "education"
        if amenity in ("hospital", "clinic", "pharmacy", "doctors"):
            return "healthcare"
        if amenity in ("parking", "fuel", "bus_station", "taxi"):
            return "transport"
        return "amenity"
    if tags.get("shop"):
        return "shop"
    if tags.get("tourism"):
        return "tourism"
    if tags.get("leisure"):
        return "leisure"
    if tags.get("office"):
        return "office"
    if tags.get("building"):
        building = tags["building"]
        if building in ("university", "school", "college"):
            return "education"
        if building in ("residential", "apartments", "dormitory"):
            return "residential"
        return "building"
    return "other"


def fetch_pois(
    lat: float,
    lng: float,
    radius_m: float = DEFAULT_RADIUS_M,
) -> list[dict]:
    """
    Fetch POIs from Overpass API for the area around (lat, lng).

    Args:
        lat:      Center latitude (decimal degrees).
        lng:      Center longitude (decimal degrees).
        radius_m: Search radius in metres (default 1000).

    Returns:
        List of POI dicts with keys: name, lat, lng, category
    """
    bbox = compute_bounding_box(lat, lng, radius_m)

    print(f"Querying Overpass API...")
    print(f"  Center:  ({lat}, {lng})")
    print(f"  Radius:  {radius_m} m")
    print(f"  Bbox:    lat [{bbox[0]:.6f}, {bbox[2]:.6f}], lng [{bbox[1]:.6f}, {bbox[3]:.6f}]")

    query = build_overpass_query(bbox)

    # Try each Overpass mirror with retries
    last_error = None
    data = None

    for url_idx, overpass_url in enumerate(OVERPASS_URLS):
        server_name = overpass_url.split("//")[1].split("/")[0]
        print(f"  Trying server: {server_name}...")

        for attempt in range(2):  # 2 attempts per server
            try:
                resp = requests.post(
                    overpass_url, data={"data": query}, headers=OVERPASS_HEADERS, timeout=90
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.exceptions.HTTPError, requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt == 0:
                    print(f"    Error: {e}. Retrying in 3s...")
                    time.sleep(3)
                else:
                    print(f"    Server {server_name} failed. Trying next mirror...")

        if data is not None:
            break

    if data is None:
        raise RuntimeError(
            f"All Overpass API servers failed. Last error: {last_error}"
        )

    elements = data.get("elements", [])
    print(f"  Raw elements returned: {len(elements)}")

    pois = []
    seen = set()  # (name, round(lat,5), round(lng,5)) for dedup

    for el in elements:
        tags = el.get("tags", {})
        name = _extract_name(tags)
        category = _extract_category(tags)

        # Get coordinates (center for ways/relations, direct for nodes)
        if "center" in el:
            el_lat = el["center"]["lat"]
            el_lng = el["center"]["lon"]
        elif el["type"] == "node":
            el_lat = el.get("lat")
            el_lng = el.get("lon")
        else:
            continue

        if el_lat is None or el_lng is None:
            continue

        # Assign fallback name for unnamed entries
        if not name or name.strip() == "":
            building_type = tags.get("building", "yes")
            if building_type == "yes":
                name = f"Building ({el['id']})"
            else:
                name = f"{building_type.replace('_', ' ').title()} ({el['id']})"

        # Filter out generic unnamed buildings/industrials — they are noise
        if _is_generic_unnamed(name):
            continue

        # Transliterate non-English characters to ASCII
        name = _transliterate_name(name)

        # Dedup key
        dedup_key = (name, round(el_lat, 5), round(el_lng, 5))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        pois.append(
            {
                "name": name.strip(),
                "lat": round(el_lat, 7),
                "lng": round(el_lng, 7),
                "category": category,
            }
        )

    # Sort by name for readability
    pois.sort(key=lambda p: p["name"])

    # Assign IDs
    for i, poi in enumerate(pois, start=1):
        poi["id"] = i

    print(f"  Filtered generic entries (Building/Industrial/etc with only IDs)")

    return pois


# ── CSV output ───────────────────────────────────────────────────────────────────


def save_csv(pois: list[dict], path: str) -> None:
    """Save POI list to CSV with id, name, lat, lng, category columns."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "lat", "lng", "category"])
        writer.writeheader()
        for poi in pois:
            writer.writerow(poi)
    print(f"Saved {len(pois)} POIs to {path}")


# ── CLI entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic POI Fetcher — fetch POIs for any location from OpenStreetMap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_poi2.py --auto
  python fetch_poi2.py --lat 24.968 --lng 121.191 --radius 1000
  python fetch_poi2.py --lat 35.6812 --lng 139.7671 --radius 500 --output tokyo.csv
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect current location via IP geolocation.",
    )
    mode_group.add_argument(
        "--lat",
        type=float,
        help="Center latitude (decimal degrees). Requires --lng.",
    )

    parser.add_argument(
        "--lng",
        type=float,
        help="Center longitude (decimal degrees). Required with --lat.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS_M,
        help=f"Search radius in metres (default {DEFAULT_RADIUS_M}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file path (default '{DEFAULT_OUTPUT}').",
    )

    args = parser.parse_args()

    # ── Resolve location ─────────────────────────────────────────────────────
    if args.auto:
        print("Detecting current location via IP geolocation...")
        try:
            loc = get_current_location()
            center_lat = loc["lat"]
            center_lng = loc["lng"]
            print(
                f"  Detected: {loc['city']}, {loc['region']}, {loc['country']}"
                f"  ({center_lat}, {center_lng})"
            )
        except RuntimeError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        if args.lng is None:
            parser.error("--lng is required when using --lat")
        center_lat = args.lat
        center_lng = args.lng
        print(f"Using manual coordinates: ({center_lat}, {center_lng})")

    # ── Fetch POIs ───────────────────────────────────────────────────────────
    print()
    pois = fetch_pois(center_lat, center_lng, args.radius)

    if not pois:
        print("\nWARNING: No POIs found. Check coordinates, radius, or network connection.")
    else:
        print(f"\nFound {len(pois)} unique POIs:")
        for p in pois[:30]:  # Show first 30
            print(f"  [{p['id']:3d}] {p['name']:45s}  ({p['lat']}, {p['lng']})  [{p['category']}]")
        if len(pois) > 30:
            print(f"  ... and {len(pois) - 30} more")

    save_csv(pois, args.output)
    print("\nDone!")
