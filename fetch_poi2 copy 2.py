"""
fetch_poi2.py — Dynamic & Flexible POI Fetcher for Any Location.

Supports two modes:

  1. OSM mode (--mode osm):
     Queries the Overpass API for buildings/POIs within a bounding box.

  2. Google Maps mode (--mode gmaps):
     Reads an existing POI CSV and enriches each POI using Playwright
     to scrape Google Maps for real names, ratings, reviews, and addresses.

Usage:
    # OSM mode — fetch raw POIs from OpenStreetMap
    python fetch_poi2.py --mode osm --auto
    python fetch_poi2.py --mode osm --lat 24.968 --lng 121.191 --radius 1000

    # Google Maps enrichment mode
    python fetch_poi2.py --mode gmaps --input poi_seed.csv --output poi_enriched.csv

    # Resume an interrupted Google Maps run
    python fetch_poi2.py --mode gmaps --input poi_seed.csv --resume

    # Limit to first N POIs (for testing)
    python fetch_poi2.py --mode gmaps --input poi_seed.csv --limit 5

    # As importable module
    from fetch_poi2 import fetch_pois, enrich_pois_gmaps
"""

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
import random
import urllib.parse
from typing import Optional

import requests
from unidecode import unidecode

# Fix Windows console encoding for non-ASCII characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Constants ────────────────────────────────────────────────────────────────────

DEFAULT_RADIUS_M = 1000  # 1 km default search radius
DEFAULT_OUTPUT = "poi_seed.csv"
DEFAULT_ENRICHED_OUTPUT = "poi_enriched.csv"
DEFAULT_PROGRESS_FILE = "_gmaps_progress.json"

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

# Google Maps scraping delays
GMAPS_MIN_DELAY = 2.0
GMAPS_MAX_DELAY = 5.0
GMAPS_PAGE_LOAD_WAIT = 3.0

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


def save_csv(pois: list[dict], path: str, fieldnames: Optional[list[str]] = None) -> None:
    """Save POI list to CSV."""
    if not pois:
        print("No POIs to save.")
        return

    if fieldnames is None:
        fieldnames = ["id", "name", "lat", "lng", "category"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for poi in pois:
            writer.writerow(poi)
    print(f"Saved {len(pois)} POIs to {path}")


# ══════════════════════════════════════════════════════════════════════════════════
# Google Maps Enrichment via Playwright
# ══════════════════════════════════════════════════════════════════════════════════


def _load_input_csv(path: str) -> list[dict]:
    """Load POIs from an input CSV file."""
    pois = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pois.append(dict(row))
    return pois


def _load_progress(progress_path: str) -> set:
    """Load the set of already-completed POI IDs from progress file."""
    if not os.path.exists(progress_path):
        return set()
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("completed_ids", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def _save_progress(progress_path: str, completed_ids: set, total: int) -> None:
    """Save progress to a JSON file for resume capability."""
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_ids": sorted(completed_ids),
                "total": total,
                "completed": len(completed_ids),
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            f,
            indent=2,
        )


def _create_playwright_browser(headless: bool = True):
    """
    Create a Playwright browser instance with stealth settings.

    Returns:
        (playwright, browser, context, page) tuple
    """
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()

    browser = pw.chromium.launch(
        headless=headless,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ],
    )

    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        locale="en",
        timezone_id="Asia/Taipei",
        geolocation=None,
        permissions=[],
    )

    # Anti-detection: override navigator.webdriver
    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en', 'zh-TW'] });
        window.chrome = { runtime: {} };
    """)

    page = context.new_page()
    # Set English headers so hours/categories are in English
    page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
    return pw, browser, context, page


def _dismiss_consent(page) -> None:
    """Try to dismiss Google consent/cookie dialogs."""
    consent_selectors = [
        "button[aria-label*='Accept']",
        "button[aria-label*='accept']",
        "form[action*='consent'] button",
        "button:has-text('Accept all')",
        "button:has-text('I agree')",
        "button:has-text('Accept')",
    ]
    for selector in consent_selectors:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=1000):
                btn.click()
                page.wait_for_timeout(1000)
                return
        except Exception:
            continue


def _haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the distance in metres between two GPS coordinates
    using the Haversine formula.
    """
    R = 6_371_000  # Earth radius in metres
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _validate_result_location(page, input_lat: float, input_lng: float, max_distance_m: float = 500) -> bool:
    """
    Check that the currently displayed Google Maps result is within
    max_distance_m of the input coordinates. Extracts coordinates from
    the page URL.

    Returns True if valid (close enough), False if wrong location.
    """
    try:
        current_url = page.url
        # Google Maps URLs contain coordinates like /@24.965,121.193,20z
        # or /place/.../@24.965,121.193
        coords_match = re.search(r'@([-\d.]+),([-\d.]+)', current_url)
        if coords_match:
            found_lat = float(coords_match.group(1))
            found_lng = float(coords_match.group(2))
            distance = _haversine_distance(input_lat, input_lng, found_lat, found_lng)
            if distance > max_distance_m:
                return False
    except Exception:
        pass  # If we can't validate, assume OK
    return True


def _try_get_place_name(page) -> str | None:
    """Try to extract the place name from the Google Maps place panel.
    Rejects coordinate strings (e.g. '24°57'54.3"N') as these indicate
    a dropped pin, not an actual business."""
    place_name_selectors = [
        "h1.DUwDvf",           # Main place name heading
        "h1.fontHeadlineLarge", # Alternative heading
        "div.lMbq3e h1",       # Nested heading
    ]
    for selector in place_name_selectors:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=2000):
                text = el.inner_text(timeout=2000)
                if text and text.strip():
                    text = text.strip()
                    # Reject coordinate strings like "24°57'54.3"N 121°11'27.1"E"
                    if re.search(r'\d+°\d+', text):
                        continue
                    # Reject pure coordinate labels
                    if re.match(r'^[\d.,\s°\'"NSEW-]+$', text):
                        continue
                    return text
        except Exception:
            continue
    return None


def _extract_hours(page) -> dict:
    """
    Extract opening hours from the Google Maps place panel.

    Returns:
        dict with 'hours_status' (e.g. 'Open', 'Closed') and
        'hours_detail' (e.g. 'Mon: 8AM-10PM; Tue: 8AM-10PM; ...')
    """
    result = {"hours_status": "", "hours_detail": ""}

    # ── Current status (Open/Closed) ─────────────────────────────────
    status_selectors = [
        "span.ZDu9vd span",              # "Open" / "Closed" text
        "div.o0JLce span.ZDu9vd",        # Status in hours section
        "span.OqCZI",                     # Operating status
    ]
    for selector in status_selectors:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=1000):
                text = el.inner_text(timeout=1000).strip()
                if text:
                    result["hours_status"] = text
                    break
        except Exception:
            continue

    # ── Try to expand the hours section ──────────────────────────────
    hours_button_selectors = [
        "button[data-item-id='oh']",       # Opening hours button
        "div[data-item-id='oh']",          # Hours container
        "button[aria-label*='hour']",
        "button[aria-label*='Hour']",
    ]

    # First try to get hours from aria-label (often has full schedule)
    for selector in hours_button_selectors:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=1000):
                aria = el.get_attribute("aria-label") or ""
                if aria and len(aria) > 10:
                    # aria-label often contains the full schedule like:
                    # "Hours: Monday, 8 AM to 10 PM; Tuesday, 8 AM to 10 PM; ..."
                    result["hours_detail"] = aria.strip()
                    break

                # Try clicking to expand
                el.click()
                page.wait_for_timeout(1000)
                break
        except Exception:
            continue

    # ── Extract expanded hours table ─────────────────────────────────
    if not result["hours_detail"]:
        hours_table_selectors = [
            "table.eK4R0e",                 # Hours table
            "table.WgFkxc",                 # Alternative hours table
            "div.t39EBf",                   # Hours rows container
        ]
        for selector in hours_table_selectors:
            try:
                el = page.locator(selector).first
                if el.is_visible(timeout=1000):
                    text = el.inner_text(timeout=2000).strip()
                    if text:
                        # Convert multi-line table to semicolon-separated
                        lines = [line.strip() for line in text.split("\n") if line.strip()]
                        result["hours_detail"] = "; ".join(lines)
                        break
            except Exception:
                continue

    return result


def _extract_description_and_review(page) -> dict:
    """
    Extract editorial description snippet and/or first review text
    from the Google Maps place panel.

    Returns:
        dict with 'description' and 'review_snippet'
    """
    result = {"description": "", "review_snippet": ""}

    # ── Editorial description / About section ────────────────────────
    desc_selectors = [
        "div.PYvSYb span",                 # Editorial summary
        "div.WeS02d div",                  # Description text
        "div[data-attrid='description'] span",
    ]
    # Service option keywords that Google puts in the description area
    _SERVICE_KEYWORDS = {
        "dine-in", "takeaway", "delivery", "takeout", "drive-through",
        "in-store shopping", "curbside pickup", "in-store pickup",
        "no-contact delivery", "makan di tempat", "bawa pulang",
        "pesan antar", "belanja di toko", "ambil di toko",
    }

    def _is_service_options_only(text: str) -> bool:
        """Check if text is composed only of Google service option labels."""
        # Strip bullets, dots, pipes, commas, spaces
        cleaned = re.sub(r'[\s·\u00b7|,]+', ' ', text).strip()
        if not cleaned:
            return True
        # Split into words and check if each segment is a service keyword
        segments = [s.strip().lower() for s in cleaned.split('  ') if s.strip()]
        if not segments:
            segments = [cleaned.lower()]
        return all(seg in _SERVICE_KEYWORDS for seg in segments)

    for selector in desc_selectors:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=1000):
                text = el.inner_text(timeout=1000).strip()
                if text and len(text) > 10:
                    # Clean up multi-line service lists
                    lines = [l.strip() for l in text.split('\n') if l.strip() and l.strip() != '\u00b7']
                    clean = ' '.join(lines)
                    # Skip if it's only service option labels
                    if _is_service_options_only(clean):
                        continue
                    if len(clean) > 10:
                        result["description"] = clean[:300]
                        break
        except Exception:
            continue

    # ── First review snippet ─────────────────────────────────────────
    review_selectors = [
        "div.MyEned span.wiI7pd",          # Review text span
        "span.wiI7pd",                     # Review text (broader)
        "div.GHT2ce span.wiI7pd",          # Review in review card
    ]
    for selector in review_selectors:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=1500):
                text = el.inner_text(timeout=1000).strip()
                if text and len(text) > 5:
                    result["review_snippet"] = text[:200]  # Cap at 200 chars
                    break
        except Exception:
            continue

    return result


def _scrape_poi_details(page, lat: float, lng: float, name: str, category: str = "") -> dict:
    """
    Navigate to a POI on Google Maps and extract detailed information.

    Strategy (fixes wrong-location bug):
    1. Navigate to coordinates first to set the map viewport
    2. Search for the POI name in the search box (viewport-constrained)
    3. If no result, fall back to nearby category search
    4. Validate result is within 500m of input coordinates
    5. Extract: name, category, rating, reviews, address, hours, description

    Returns:
        dict with all extracted fields, or empty values on failure.
    """
    result = {
        "name_local": None,
        "category_gmaps": None,
        "rating": None,
        "review_count": None,
        "address": None,
        "hours_status": None,
        "hours_detail": None,
        "description": None,
        "review_snippet": None,
    }

    try:
        # ── Strategy 1: Search by name at coordinates ────────────────
        # Use a single URL that combines POI name with coordinate viewport.
        # The ,20z zoom level constrains results to a very tight area
        # around the coordinates. Each page.goto is a fresh request,
        # so there is no stale search state between POIs.
        encoded_name = urllib.parse.quote(name)
        search_url = f"https://www.google.com/maps/search/{encoded_name}/@{lat},{lng},20z?hl=en"
        page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(int(GMAPS_PAGE_LOAD_WAIT * 1000))

        # Dismiss consent dialog if present
        _dismiss_consent(page)
        page.wait_for_timeout(1500)

        # Check if we landed on a place panel directly
        has_place_panel = False
        place_name = _try_get_place_name(page)
        has_place_panel = place_name is not None

        # If search results list, click the first result
        if not has_place_panel:
            result_selectors = [
                "a.hfpxzc",
                "div.Nv2PK a",
                "div[role='feed'] a[href*='maps/place']",
            ]
            for selector in result_selectors:
                try:
                    first_result = page.locator(selector).first
                    if first_result.is_visible(timeout=2000):
                        first_result.click()
                        page.wait_for_timeout(int(GMAPS_PAGE_LOAD_WAIT * 1000))
                        break
                except Exception:
                    continue

            place_name = _try_get_place_name(page)
            has_place_panel = place_name is not None

        # Validate location if we found something
        if has_place_panel:
            if not _validate_result_location(page, lat, lng, max_distance_m=500):
                has_place_panel = False
                place_name = None

        if not has_place_panel:
            # ── Strategy 2: Nearby category search ───────────────────
            # Search by CATEGORY (not name!) near the coordinates.
            search_term = category.replace("_", " ") if category and category != "other" else "restaurant"
            search_url = f"https://www.google.com/maps/search/{search_term}/@{lat},{lng},20z?hl=en"
            page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(int(GMAPS_PAGE_LOAD_WAIT * 1000))

            # Click first search result
            result_selectors = [
                "a.hfpxzc",
                "div.Nv2PK a",
                "div[role='feed'] a[href*='maps/place']",
            ]
            for selector in result_selectors:
                try:
                    first_result = page.locator(selector).first
                    if first_result.is_visible(timeout=2000):
                        first_result.click()
                        page.wait_for_timeout(int(GMAPS_PAGE_LOAD_WAIT * 1000))
                        break
                except Exception:
                    continue

            place_name = _try_get_place_name(page)
            has_place_panel = place_name is not None

            # Validate again
            if has_place_panel and not _validate_result_location(page, lat, lng, max_distance_m=300):
                has_place_panel = False
                place_name = None

        if not has_place_panel:
            return result

        result["name_local"] = place_name

        # ── Extract category ─────────────────────────────────────────
        category_selectors = [
            "button.DkEaL",
            "span.DkEaL",
            "button[jsaction*='category']",
            "div.LBgpJf button",
        ]
        for selector in category_selectors:
            try:
                el = page.locator(selector).first
                if el.is_visible(timeout=1000):
                    text = el.inner_text(timeout=1000)
                    if text and text.strip():
                        result["category_gmaps"] = text.strip()
                        break
            except Exception:
                continue

        # ── Extract rating ───────────────────────────────────────────
        rating_selectors = [
            "div.F7nice span[aria-hidden='true']",
            "span.ceNzKf",
            "div.fontDisplayLarge",
        ]
        for selector in rating_selectors:
            try:
                el = page.locator(selector).first
                if el.is_visible(timeout=1000):
                    text = el.inner_text(timeout=1000).strip().replace(",", ".")
                    if text and re.match(r"^\d+\.?\d*$", text):
                        result["rating"] = float(text)
                        break
            except Exception:
                continue

        # ── Extract review count ─────────────────────────────────────
        review_count_selectors = [
            "div.F7nice span span",
            "span.F7nice span[aria-label]",
            "button[aria-label*='review']",
            "button[aria-label*='Review']",
        ]
        for selector in review_count_selectors:
            try:
                elements = page.locator(selector).all()
                for el in elements:
                    try:
                        text = el.inner_text(timeout=500)
                        aria = el.get_attribute("aria-label") or ""
                        combined = text + " " + aria

                        match = re.search(
                            r"[\(]?([\d,\.]+)[\)]?\s*(?:review|ulasan|評論|件|個評論)?",
                            combined, re.I,
                        )
                        if match:
                            count_str = match.group(1).replace(",", "").replace(".", "")
                            if count_str.isdigit() and int(count_str) > 0:
                                result["review_count"] = int(count_str)
                                break
                    except Exception:
                        continue
                if result["review_count"]:
                    break
            except Exception:
                continue

        # ── Extract address ──────────────────────────────────────────
        address_selectors = [
            "button[data-item-id='address'] div.fontBodyMedium",
            "button[aria-label*='Address'] div.fontBodyMedium",
            "button[data-tooltip='Copy address'] div.Io6YTe",
            "div[data-attrid='address'] span",
        ]
        for selector in address_selectors:
            try:
                el = page.locator(selector).first
                if el.is_visible(timeout=1000):
                    text = el.inner_text(timeout=1000)
                    if text and text.strip() and len(text.strip()) > 3:
                        result["address"] = text.strip()
                        break
            except Exception:
                continue

        # ── Extract opening hours ────────────────────────────────────
        hours_data = _extract_hours(page)
        result["hours_status"] = hours_data.get("hours_status", "")
        result["hours_detail"] = hours_data.get("hours_detail", "")

        # ── Extract description and review snippet ───────────────────
        desc_data = _extract_description_and_review(page)
        result["description"] = desc_data.get("description", "")
        result["review_snippet"] = desc_data.get("review_snippet", "")

    except Exception:
        # Graceful per-POI failure
        pass

    return result


def enrich_pois_gmaps(
    input_csv: str = DEFAULT_OUTPUT,
    output_csv: str = DEFAULT_ENRICHED_OUTPUT,
    headless: bool = True,
    resume: bool = False,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Enrich POIs from input CSV by scraping Google Maps via Playwright.

    Args:
        input_csv:  Path to input POI CSV (from Overpass fetch).
        output_csv: Path for enriched output CSV.
        headless:   Run browser in headless mode.
        resume:     If True, skip POIs that were already completed.
        limit:      If set, only process the first N POIs.

    Returns:
        List of enriched POI dicts.
    """
    # Load input
    pois = _load_input_csv(input_csv)
    if not pois:
        print("No POIs found in input CSV.")
        return []

    if limit:
        pois = pois[:limit]

    # Progress tracking
    base_dir = os.path.dirname(os.path.abspath(output_csv))
    progress_path = os.path.join(
        base_dir,
        os.path.splitext(os.path.basename(output_csv))[0] + DEFAULT_PROGRESS_FILE,
    )

    completed_ids = set()
    if resume:
        completed_ids = _load_progress(progress_path)
        if completed_ids:
            print(f"Resuming: {len(completed_ids)}/{len(pois)} already completed")

    print(f"Loaded {len(pois)} POIs from {input_csv}")
    to_process = len(pois) - len(completed_ids)
    print(f"POIs to process: {to_process}")
    print(f"Mode: {'headless' if headless else 'visible'}")
    est_minutes = (to_process * 5) / 60
    print(f"Estimated time: ~{est_minutes:.0f} minutes")
    print()

    # Load existing enriched data if resuming
    enriched_data = {}
    if resume and os.path.exists(output_csv):
        try:
            existing = _load_input_csv(output_csv)
            for row in existing:
                enriched_data[row.get("id", "")] = row
        except Exception:
            pass

    # Create browser
    print("Launching Playwright browser...")
    pw, browser, context, page = _create_playwright_browser(headless=headless)

    enriched = []
    succeeded = 0
    failed = 0

    try:
        for idx, poi in enumerate(pois, start=1):
            poi_id = poi.get("id", str(idx))
            name = poi.get("name", "Unknown")
            lat = float(poi.get("lat", 0))
            lng = float(poi.get("lng", 0))

            # Skip if already completed (resume mode)
            if str(poi_id) in completed_ids:
                # Use cached data
                if str(poi_id) in enriched_data:
                    enriched.append(enriched_data[str(poi_id)])
                else:
                    enriched.append(dict(poi))
                continue

            print(f"  [{idx}/{len(pois)}] {name[:40]:40s}...", end=" ", flush=True)

            try:
                category = poi.get("category", "")
                details = _scrape_poi_details(page, lat, lng, name, category)

                # Build enriched POI
                enriched_poi = dict(poi)
                enriched_poi["name_local"] = details.get("name_local") or ""
                enriched_poi["name_en"] = _transliterate_name(
                    details.get("name_local") or name
                )
                enriched_poi["category_gmaps"] = details.get("category_gmaps") or ""
                enriched_poi["rating"] = details.get("rating") or ""
                enriched_poi["review_count"] = details.get("review_count") or ""
                enriched_poi["address"] = details.get("address") or ""
                enriched_poi["hours_status"] = details.get("hours_status") or ""
                enriched_poi["hours_detail"] = details.get("hours_detail") or ""
                enriched_poi["description"] = details.get("description") or ""
                enriched_poi["review_snippet"] = details.get("review_snippet") or ""

                enriched.append(enriched_poi)
                completed_ids.add(str(poi_id))
                succeeded += 1

                # Status output
                rating_str = f"★ {details['rating']}" if details.get("rating") else "no rating"
                name_str = details.get("name_local", "")[:20] if details.get("name_local") else "—"
                count_str = f"{details['review_count']} reviews" if details.get("review_count") else ""
                hours_str = f"[{details.get('hours_status', '')}]" if details.get("hours_status") else ""
                print(f"{rating_str} {count_str} {hours_str} [{name_str}]")

            except Exception as e:
                print(f"FAILED ({e})")
                enriched_poi = dict(poi)
                enriched_poi["name_local"] = ""
                enriched_poi["name_en"] = name
                enriched_poi["category_gmaps"] = ""
                enriched_poi["rating"] = ""
                enriched_poi["review_count"] = ""
                enriched_poi["address"] = ""
                enriched_poi["hours_status"] = ""
                enriched_poi["hours_detail"] = ""
                enriched_poi["description"] = ""
                enriched_poi["review_snippet"] = ""
                enriched.append(enriched_poi)
                failed += 1

            # Save progress after each POI
            _save_progress(progress_path, completed_ids, len(pois))

            # Incremental save every 10 POIs
            if idx % 10 == 0:
                _save_enriched_csv(enriched, output_csv)

            # Random delay between POIs
            if idx < len(pois):
                delay = random.uniform(GMAPS_MIN_DELAY, GMAPS_MAX_DELAY)
                time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving progress ({len(completed_ids)}/{len(pois)} completed)...")
    finally:
        # Final save
        _save_enriched_csv(enriched, output_csv)
        _save_progress(progress_path, completed_ids, len(pois))

        # Cleanup
        try:
            page.close()
            context.close()
            browser.close()
            pw.stop()
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"Results: {succeeded} succeeded, {failed} failed, {len(completed_ids)}/{len(pois)} total")
    print(f"Output:  {output_csv}")
    if len(completed_ids) < len(pois):
        print(f"Resume:  Run again with --resume to continue")
    print(f"{'='*60}")

    return enriched


def _save_enriched_csv(enriched: list[dict], path: str) -> None:
    """Save enriched POI data to CSV."""
    if not enriched:
        return

    fieldnames = [
        "id", "name", "name_local", "name_en",
        "lat", "lng",
        "category", "category_gmaps",
        "rating", "review_count",
        "address",
        "hours_status", "hours_detail",
        "description", "review_snippet",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for poi in enriched:
            writer.writerow(poi)


# ── CLI entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic POI Fetcher — OSM fetch or Google Maps enrichment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  osm    — Fetch POIs from OpenStreetMap Overpass API (default)
  gmaps  — Enrich existing POI CSV via Google Maps Playwright scraping

Examples:
  python fetch_poi2.py --mode osm --auto
  python fetch_poi2.py --mode osm --lat 24.968 --lng 121.191 --radius 1000
  python fetch_poi2.py --mode gmaps --input poi_seed.csv --output poi_enriched.csv
  python fetch_poi2.py --mode gmaps --input poi_seed.csv --limit 5
  python fetch_poi2.py --mode gmaps --input poi_seed.csv --resume
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["osm", "gmaps"],
        default="osm",
        help="Fetch mode: 'osm' for Overpass API, 'gmaps' for Google Maps enrichment.",
    )

    # OSM mode arguments
    osm_group = parser.add_argument_group("OSM mode options")
    osm_group.add_argument("--auto", action="store_true", help="Auto-detect location via IP.")
    osm_group.add_argument("--lat", type=float, help="Center latitude.")
    osm_group.add_argument("--lng", type=float, help="Center longitude.")
    osm_group.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M, help=f"Search radius in metres (default {DEFAULT_RADIUS_M}).")

    # Google Maps mode arguments
    gmaps_group = parser.add_argument_group("Google Maps mode options")
    gmaps_group.add_argument("--input", type=str, default=DEFAULT_OUTPUT, help=f"Input POI CSV (default '{DEFAULT_OUTPUT}').")
    gmaps_group.add_argument("--limit", type=int, help="Only process the first N POIs (for testing).")
    gmaps_group.add_argument("--resume", action="store_true", help="Resume from last progress checkpoint.")
    gmaps_group.add_argument("--visible", action="store_true", help="Show browser window (for debugging).")

    # Shared arguments
    parser.add_argument("--output", type=str, help="Output CSV file path.")

    args = parser.parse_args()

    if args.mode == "osm":
        # ── OSM / Overpass mode ──────────────────────────────────────────────
        output_path = args.output or DEFAULT_OUTPUT

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
        elif args.lat is not None:
            if args.lng is None:
                parser.error("--lng is required when using --lat")
            center_lat = args.lat
            center_lng = args.lng
            print(f"Using manual coordinates: ({center_lat}, {center_lng})")
        else:
            parser.error("OSM mode requires --auto or --lat/--lng")

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

        save_csv(pois, output_path)
        print("\nDone!")

    elif args.mode == "gmaps":
        # ── Google Maps enrichment mode ──────────────────────────────────────
        output_path = args.output or DEFAULT_ENRICHED_OUTPUT
        input_path = args.input

        if not os.path.exists(input_path):
            print(f"ERROR: Input file '{input_path}' not found.")
            print("Run with --mode osm first to generate the POI seed CSV.")
            sys.exit(1)

        print("=" * 60)
        print("Google Maps POI Enrichment via Playwright")
        print("=" * 60)

        enrich_pois_gmaps(
            input_csv=input_path,
            output_csv=output_path,
            headless=not args.visible,
            resume=args.resume,
            limit=args.limit,
        )

        print("\nDone!")
