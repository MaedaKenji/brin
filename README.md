# Surrounding Awareness — Dynamic Location API

A state-of-the-art API service merging computer vision (YOLO) and GPS/altitude spatial context to provide intelligent surrounding awareness. Works with **any GPS location worldwide** — fetch POIs from OpenStreetMap, auto-detect indoor/outdoor context, and analyse images or videos with a single command.

---

## 🚀 Key Features

* **Dual-Context Awareness:** Merges visual predictions with spatial coordinates (GPS/Altitude).
* **Dynamic Multi-Location Support:** Fetch POIs for any GPS coordinate via OpenStreetMap — not locked to a single campus or city.
* **Dynamic Model Routing:** Automatically chooses the optimal YOLO model (`indoor.pt` or `outdoor.pt`) based on proximity to points of interest, with manual override capabilities.
* **Intelligent Post-Processing:** Uses per-class Non-Maximum Suppression (NMS) and custom door-specific containment rules to keep detections precise.
* **Directional Reasoning:** Detects doors and estimates their relative direction (left, front, right).
* **Rich Dashboard:** A built-in web frontend for uploading media and visualizing detections in real-time.

---

## 📖 Detailed Documentation

For full details regarding the architecture, endpoint specifications, prerequisites, step-by-step setup guides, and running instructions, please refer to the walkthrough document:

👉 **[walkthrough.md](walkthrough.md)**

---

## 🛠️ Quick Start

```bash
# 1. Clone and install
git clone <repository-url>
cd brin
pip install -r requirements.txt

# 2. Launch (single command)
uvicorn final4:app --reload --port 8000
```

Open **http://localhost:8000/** in your browser.

### Use Your Own Location

```bash
# Fetch POIs for your GPS coordinates (replace with yours)
curl -X POST "http://localhost:8000/api/fetch-pois" \
  -H "Content-Type: application/json" \
  -d '{"lat": 40.7128, "lng": -74.0060, "radius_m": 1000}'

# Analyse an image
curl -X POST "http://localhost:8000/api/analyze" \
  -F "image=@photo.jpg" \
  -F "lat=40.7128" \
  -F "lng=-74.0060" \
  -F "poi_csv=poi_seed.csv"
```

See the walkthrough for full API reference, architecture details, and troubleshooting.
