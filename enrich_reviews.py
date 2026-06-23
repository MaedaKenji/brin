"""
enrich_reviews.py — Fase 2: Google Maps Review Enrichment via Selenium.

Takes a POI CSV (output from fetch_poi.py) and enriches each POI with
Google Maps review data using Selenium browser automation.

This step is OPTIONAL — the rest of the pipeline works without it.

Usage:
    # Basic enrichment (headless)
    python enrich_reviews.py --input poi_seed.csv

    # With visible browser for debugging
    python enrich_reviews.py --input poi_seed.csv --visible --max-reviews 5

    # Custom output paths
    python enrich_reviews.py --input poi_seed.csv --output poi_enriched.csv --reviews reviews.json

    # As importable module
    from enrich_reviews import enrich_pois
    results = enrich_pois("poi_seed.csv", max_reviews=10)

WARNING: Google Maps actively blocks automated scraping. This module uses
best-effort selectors and anti-detection measures, but may break when
Google updates their site. Failures are handled gracefully per-POI.
"""

import argparse
import csv
import json
import io
import os
import random
import sys
import time
from typing import Optional

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Constants ────────────────────────────────────────────────────────────────────

DEFAULT_INPUT = "poi_seed.csv"
DEFAULT_OUTPUT = "poi_enriched.csv"
DEFAULT_REVIEWS_JSON = "reviews.json"
DEFAULT_MAX_REVIEWS = 20

GMAPS_SEARCH_URL = "https://www.google.com/maps/search/"

# Delay ranges (seconds) to avoid rate limiting
MIN_DELAY = 2.0
MAX_DELAY = 5.0
PAGE_LOAD_WAIT = 4.0

# ── Selenium driver setup ────────────────────────────────────────────────────────


def create_driver(headless: bool = True):
    """
    Create a Chrome WebDriver with anti-detection options.

    Args:
        headless: Run in headless mode (default True).

    Returns:
        selenium.webdriver.Chrome instance.

    Raises:
        ImportError: If selenium or webdriver-manager is not installed.
        RuntimeError: If Chrome/Chromium is not available.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except ImportError:
        raise ImportError(
            "selenium is required for review enrichment. "
            "Install it with: pip install selenium webdriver-manager"
        )

    try:
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        raise ImportError(
            "webdriver-manager is required. "
            "Install it with: pip install webdriver-manager"
        )

    options = Options()

    if headless:
        options.add_argument("--headless=new")

    # Anti-detection measures
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create Chrome WebDriver: {e}\n"
            "Make sure Chrome or Chromium is installed on your system."
        )

    # Additional anti-detection: remove webdriver property
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        },
    )

    driver.implicitly_wait(5)
    return driver


# ── Google Maps interaction ──────────────────────────────────────────────────────


def _random_delay(min_s: float = MIN_DELAY, max_s: float = MAX_DELAY) -> None:
    """Sleep for a random duration to mimic human behavior."""
    time.sleep(random.uniform(min_s, max_s))


def search_poi_on_gmaps(
    driver, lat: float, lng: float, name: str
) -> Optional[str]:
    """
    Navigate to a POI on Google Maps using coordinates.

    Args:
        driver: Selenium WebDriver instance.
        lat:    POI latitude.
        lng:    POI longitude.
        name:   POI name (used in search query for better matching).

    Returns:
        The current URL after navigation, or None on failure.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    # Use coordinate-based URL for precise location
    url = f"https://www.google.com/maps/@{lat},{lng},18z"
    driver.get(url)
    time.sleep(PAGE_LOAD_WAIT)

    # Try to dismiss cookie consent if it appears
    try:
        consent_btn = driver.find_element(
            By.CSS_SELECTOR, "button[aria-label*='Accept'], form[action*='consent'] button"
        )
        consent_btn.click()
        time.sleep(1)
    except Exception:
        pass

    # Now search for the specific POI name nearby
    try:
        search_box = WebDriverWait(driver, 8).until(
            EC.presence_of_element_located((By.ID, "searchboxinput"))
        )
        search_box.clear()
        search_box.send_keys(f"{name}")
        search_box.send_keys("\n")
        time.sleep(PAGE_LOAD_WAIT)
    except Exception:
        pass

    return driver.current_url


def extract_reviews(driver, max_reviews: int = DEFAULT_MAX_REVIEWS) -> dict:
    """
    Extract review data from the current Google Maps POI page.

    Args:
        driver:      Selenium WebDriver instance (already on a POI page).
        max_reviews: Maximum number of individual reviews to extract.

    Returns:
        dict with keys:
            rating:       float or None — overall rating
            review_count: int or None   — total number of reviews
            reviews:      list of dicts — individual review data
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.action_chains import ActionChains

    result = {
        "rating": None,
        "review_count": None,
        "reviews": [],
    }

    # ── Extract overall rating ───────────────────────────────────────────
    try:
        # Try multiple selectors for rating
        rating_selectors = [
            "div.fontDisplayLarge",
            "span.ceNzKf",  # Numeric rating
            "div.F7nice span[aria-hidden]",
        ]
        for selector in rating_selectors:
            try:
                el = driver.find_element(By.CSS_SELECTOR, selector)
                text = el.text.strip().replace(",", ".")
                if text:
                    result["rating"] = float(text)
                    break
            except Exception:
                continue
    except Exception:
        pass

    # ── Extract review count ─────────────────────────────────────────────
    try:
        count_selectors = [
            "div.fontBodySmall span",  # e.g., "(1,234 reviews)"
            "button[aria-label*='review']",
            "span.F7nice span[aria-label]",
        ]
        for selector in count_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for el in elements:
                    text = el.text or el.get_attribute("aria-label") or ""
                    # Look for patterns like "1,234 reviews" or "1234 ulasan"
                    import re

                    match = re.search(r"([\d,.]+)\s*(?:review|ulasan|評論|件)", text, re.I)
                    if match:
                        count_str = match.group(1).replace(",", "").replace(".", "")
                        result["review_count"] = int(count_str)
                        break
                if result["review_count"]:
                    break
            except Exception:
                continue
    except Exception:
        pass

    # ── Try to open the reviews tab ──────────────────────────────────────
    try:
        review_tab_selectors = [
            "button[aria-label*='Review']",
            "button[aria-label*='review']",
            "button[data-tab-index='1']",
            "div.RWPxGd button",
        ]
        for selector in review_tab_selectors:
            try:
                btn = driver.find_element(By.CSS_SELECTOR, selector)
                btn.click()
                time.sleep(2)
                break
            except Exception:
                continue
    except Exception:
        pass

    # ── Scroll and extract individual reviews ────────────────────────────
    try:
        # Find the scrollable reviews container
        scroll_selectors = [
            "div.m6QErb.DxyBCb.kA9KIf.dS8AEf",
            "div.m6QErb.DxyBCb",
            "div[role='main'] div.m6QErb",
        ]
        scrollable = None
        for selector in scroll_selectors:
            try:
                scrollable = driver.find_element(By.CSS_SELECTOR, selector)
                break
            except Exception:
                continue

        if scrollable:
            # Scroll to load more reviews
            scroll_attempts = 0
            max_scroll_attempts = max_reviews // 3 + 5

            while scroll_attempts < max_scroll_attempts:
                driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight", scrollable
                )
                time.sleep(1.5)
                scroll_attempts += 1

                # Check how many reviews we have
                review_elements = driver.find_elements(
                    By.CSS_SELECTOR, "div.jftiEf, div[data-review-id]"
                )
                if len(review_elements) >= max_reviews:
                    break

        # Extract individual reviews
        review_elements = driver.find_elements(
            By.CSS_SELECTOR, "div.jftiEf, div[data-review-id]"
        )

        for rev_el in review_elements[:max_reviews]:
            review = {}

            # Author name
            try:
                author_el = rev_el.find_element(By.CSS_SELECTOR, "div.d4r55, button.WEBjve")
                review["author"] = author_el.text.strip()
            except Exception:
                review["author"] = None

            # Star rating
            try:
                stars_el = rev_el.find_element(By.CSS_SELECTOR, "span.kvMYJc")
                aria = stars_el.get_attribute("aria-label") or ""
                import re

                match = re.search(r"(\d+)", aria)
                review["rating"] = int(match.group(1)) if match else None
            except Exception:
                review["rating"] = None

            # Review text
            try:
                # Try to click "More" button if exists
                try:
                    more_btn = rev_el.find_element(
                        By.CSS_SELECTOR, "button.w8nwRe.kyuRq"
                    )
                    more_btn.click()
                    time.sleep(0.5)
                except Exception:
                    pass

                text_el = rev_el.find_element(
                    By.CSS_SELECTOR, "span.wiI7pd, div.MyEned span"
                )
                review["text"] = text_el.text.strip()
            except Exception:
                review["text"] = ""

            # Review date
            try:
                date_el = rev_el.find_element(By.CSS_SELECTOR, "span.rsqaWe")
                review["date"] = date_el.text.strip()
            except Exception:
                review["date"] = None

            if review.get("author") or review.get("text"):
                result["reviews"].append(review)

    except Exception:
        pass

    return result


# ── Batch enrichment pipeline ────────────────────────────────────────────────────


def enrich_pois(
    input_csv: str = DEFAULT_INPUT,
    output_csv: str = DEFAULT_OUTPUT,
    reviews_json: str = DEFAULT_REVIEWS_JSON,
    max_reviews: int = DEFAULT_MAX_REVIEWS,
    headless: bool = True,
) -> list[dict]:
    """
    Batch-enrich POIs with Google Maps review data.

    Reads POIs from input_csv, visits each on Google Maps, extracts reviews,
    and saves enriched data to output_csv + reviews_json.

    Args:
        input_csv:    Path to POI CSV (from fetch_poi.py).
        output_csv:   Path for enriched POI CSV output.
        reviews_json: Path for detailed review JSON output.
        max_reviews:  Max reviews to extract per POI.
        headless:     Run browser in headless mode.

    Returns:
        List of enriched POI dicts.
    """
    # Load input POIs
    pois = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pois.append(row)

    if not pois:
        print("No POIs found in input CSV.")
        return []

    print(f"Loaded {len(pois)} POIs from {input_csv}")
    print(f"Starting Google Maps review enrichment (max {max_reviews} reviews per POI)...")
    print(f"Mode: {'headless' if headless else 'visible'}")
    print()

    # Create browser driver
    driver = create_driver(headless=headless)

    enriched = []
    all_reviews = {}

    try:
        for idx, poi in enumerate(pois, start=1):
            name = poi.get("name", "Unknown")
            lat = float(poi.get("lat", 0))
            lng = float(poi.get("lng", 0))
            poi_id = poi.get("id", idx)

            print(f"  [{idx}/{len(pois)}] {name}...", end=" ", flush=True)

            try:
                # Navigate to the POI
                search_poi_on_gmaps(driver, lat, lng, name)

                # Extract reviews
                review_data = extract_reviews(driver, max_reviews)

                # Enrich the POI entry
                enriched_poi = dict(poi)
                enriched_poi["rating"] = review_data["rating"]
                enriched_poi["review_count"] = review_data["review_count"]
                enriched.append(enriched_poi)

                # Store detailed reviews
                all_reviews[str(poi_id)] = {
                    "name": name,
                    "lat": lat,
                    "lng": lng,
                    "rating": review_data["rating"],
                    "review_count": review_data["review_count"],
                    "reviews": review_data["reviews"],
                }

                rating_str = f"★ {review_data['rating']}" if review_data["rating"] else "no rating"
                count_str = f"{review_data['review_count']} reviews" if review_data["review_count"] else "no reviews"
                n_extracted = len(review_data["reviews"])
                print(f"{rating_str}, {count_str}, {n_extracted} extracted")

            except Exception as e:
                print(f"FAILED ({e})")
                enriched_poi = dict(poi)
                enriched_poi["rating"] = None
                enriched_poi["review_count"] = None
                enriched.append(enriched_poi)

            # Random delay between POIs
            if idx < len(pois):
                _random_delay()

    finally:
        driver.quit()

    # ── Save enriched CSV ────────────────────────────────────────────────
    if enriched:
        fieldnames = list(enriched[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in enriched:
                writer.writerow(row)
        print(f"\nSaved enriched POI data to {output_csv}")

    # ── Save detailed reviews JSON ───────────────────────────────────────
    if all_reviews:
        with open(reviews_json, "w", encoding="utf-8") as f:
            json.dump(all_reviews, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed reviews to {reviews_json}")

    return enriched


# ── CLI entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Google Maps Review Enrichment — enrich POI data with reviews.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enrich_reviews.py --input poi_seed.csv
  python enrich_reviews.py --input poi_seed.csv --visible --max-reviews 5
  python enrich_reviews.py --input poi_seed.csv --output enriched.csv --reviews reviews.json
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Input POI CSV (default '{DEFAULT_INPUT}').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output enriched CSV (default '{DEFAULT_OUTPUT}').",
    )
    parser.add_argument(
        "--reviews",
        type=str,
        default=DEFAULT_REVIEWS_JSON,
        help=f"Output reviews JSON (default '{DEFAULT_REVIEWS_JSON}').",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=DEFAULT_MAX_REVIEWS,
        help=f"Max reviews per POI (default {DEFAULT_MAX_REVIEWS}).",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show the browser window (for debugging). Default is headless.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' not found.")
        print("Run fetch_poi.py first to generate the POI seed CSV.")
        sys.exit(1)

    print("=" * 60)
    print("Google Maps Review Enrichment (Fase 2)")
    print("=" * 60)

    enrich_pois(
        input_csv=args.input,
        output_csv=args.output,
        reviews_json=args.reviews,
        max_reviews=args.max_reviews,
        headless=not args.visible,
    )

    print("\nDone!")
