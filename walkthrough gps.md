# GPS POI Direction Awareness System — Walkthrough

## What Was Built

A complete POI direction awareness system for NCU campus that identifies nearby buildings and classifies them by compass direction from any GPS coordinate.

## Files Created

### [fetch_poi.py]
- One-time script to fetch building POIs from OpenStreetMap Overpass API
- Collects building names, polygon centroids, and amenity points
- Cleans data: deduplicates, handles unnamed buildings, normalizes names
- Outputs `poi_buildings.csv`

### [poi_buildings.csv]
- **184 POIs** collected from NCU campus (buildings, restaurants, dormitories, labs, etc.)
- Columns: `id, name, lat, lng`

### [gps.py]
The main module containing:

| Component | Description |
|-----------|-------------|
| `load_pois()` | Loads POI data from CSV |
| `build_kdtree()` | Builds KDTree spatial index for O(log n) queries |
| `haversine()` | Great-circle distance between two GPS points |
| `bearing()` | Azimuth angle (0°=N, 90°=E, 180°=S, 270°=W) |
| `bearing_to_direction()` | Converts angle → 8-way compass sector |
| `get_nearby_pois()` | Core query: returns POIs grouped by direction |
| `query()` | Convenience wrapper with cached data |
| FastAPI app | REST API with `/direction-poi`, `/pois`, `/health` |

## Usage

### As importable module (for `inference2.py` integration)
```python
from gps import query

result = query(24.968, 121.191, radius_m=500)

# result["directions"]["North"] → list of POIs to the north
# result["nearest"] → closest building info
# result["total_pois_found"] → count
```

### As FastAPI server
```bash
uvicorn gps:app --reload --port 8000
# POST to http://localhost:8000/direction-poi
# Swagger UI at http://localhost:8000/docs
```

## Test Results

| Test | Input | Result |
|------|-------|--------|
| Center of campus | `24.968, 121.191` | 155 POIs, nearest: 依仁堂 (44.2m, NW) |
| Near Library | `24.9683, 121.1943` | POIs in all directions |
| Outside campus | `0.0, 0.0` | 0 POIs, no crash ✓ |
| FastAPI `/health` | GET | `{"status":"ok","pois_loaded":184}` ✓ |
| FastAPI `/docs` | GET | Swagger UI loads ✓ |

## API Response Format
```json
{
  "input": {"lat": 24.968, "lng": 121.191},
  "radius_m": 500,
  "directions": {
    "North": [{"name": "...", "distance_m": 120.5, "bearing": 5.2, "lat": ..., "lng": ...}],
    "South": [...],
    ...
  },
  "nearest": {"name": "依仁堂", "distance_m": 44.2, "direction": "North-West"},
  "total_pois_found": 155
}
```
