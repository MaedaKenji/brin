"""
fetch_poi.py — Fetch building POIs for National Central University (NCU) from OpenStreetMap.

Queries the Overpass API for buildings within the NCU campus bounding box,
extracts names and centroid coordinates, cleans duplicates, and saves to CSV.

Usage:
    python fetch_poi.py
"""

import csv
import sys
import io
import requests

# Fix Windows console encoding for Chinese characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── NCU campus bounding box ─────────────────────────────────────────────────────
# National Central University (中央大學), Zhongli, Taoyuan, Taiwan
MIN_LAT = 24.965
MAX_LAT = 24.972
MIN_LNG = 121.185
MAX_LNG = 121.198

OUTPUT_FILE = "poi_buildings.csv"

# ── Overpass API query ───────────────────────────────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OVERPASS_QUERY = f"""
[out:json][timeout:30];
(
  way["building"]({MIN_LAT},{MIN_LNG},{MAX_LAT},{MAX_LNG});
  relation["building"]({MIN_LAT},{MIN_LNG},{MAX_LAT},{MAX_LNG});
  node["building"]({MIN_LAT},{MIN_LNG},{MAX_LAT},{MAX_LNG});
  node["amenity"]({MIN_LAT},{MIN_LNG},{MAX_LAT},{MAX_LNG});
);
out center;
"""

HEADERS = {
    "User-Agent": "NCU-POI-Fetcher/1.0",
    "Accept": "application/json",
}


def fetch_pois():
    """Fetch building POIs from Overpass API."""
    print(f"Querying Overpass API for NCU buildings...")
    print(f"Bounding box: lat [{MIN_LAT}, {MAX_LAT}], lng [{MIN_LNG}, {MAX_LNG}]")

    resp = requests.post(OVERPASS_URL, data={"data": OVERPASS_QUERY},
                         headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    elements = data.get("elements", [])
    print(f"Raw elements returned: {len(elements)}")

    pois = []
    seen = set()  # (name, round(lat,5), round(lng,5)) for dedup

    for el in elements:
        # Get name (try multiple tags)
        tags = el.get("tags", {})
        name = (
            tags.get("name:en")
            or tags.get("name")
            or tags.get("official_name")
            or tags.get("alt_name")
            or tags.get("amenity", "").replace("_", " ").title()
            or None
        )

        # Get coordinates (center for ways/relations, direct for nodes)
        if "center" in el:
            lat = el["center"]["lat"]
            lng = el["center"]["lon"]
        elif el["type"] == "node":
            lat = el.get("lat")
            lng = el.get("lon")
        else:
            continue

        if lat is None or lng is None:
            continue

        # Skip unnamed entries that have no useful tag
        if not name or name.strip() == "":
            building_type = tags.get("building", "yes")
            if building_type == "yes":
                name = f"Building ({el['id']})"
            else:
                name = f"{building_type.replace('_', ' ').title()} ({el['id']})"

        # Dedup key
        dedup_key = (name, round(lat, 5), round(lng, 5))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        pois.append({
            "name": name.strip(),
            "lat": round(lat, 7),
            "lng": round(lng, 7),
        })

    # Sort by name for readability
    pois.sort(key=lambda p: p["name"])

    # Assign IDs
    for i, poi in enumerate(pois, start=1):
        poi["id"] = i

    return pois


def save_csv(pois, path):
    """Save POI list to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "lat", "lng"])
        writer.writeheader()
        for poi in pois:
            writer.writerow(poi)
    print(f"Saved {len(pois)} POIs to {path}")


if __name__ == "__main__":
    pois = fetch_pois()

    if not pois:
        print("WARNING: No POIs found. Check bounding box or network connection.")
    else:
        print(f"\nFound {len(pois)} unique POIs:")
        for p in pois:
            print(f"  [{p['id']:3d}] {p['name']:40s}  ({p['lat']}, {p['lng']})")

    save_csv(pois, OUTPUT_FILE)
    print("\nDone!")
