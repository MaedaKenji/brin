# Unified Surrounding Awareness API Service

A premium, state-of-the-art API service merging computer vision (YOLO) and GPS/altitude spatial context to provide intelligent surrounding awareness. It dynamically adjusts its inference model based on geographic context, tracks environment details (indoor vs. outdoor), filters redundant detections, and translates altitude measurements into floor estimates.

---

## 🚀 Key Features

* **Dual-Context Awareness:** Merges visual predictions with spatial coordinates (GPS/Altitude).
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

If you'd like to get up and running immediately:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Model Files in Root:**
   * `indoor.pt`
   * `outdoor.pt`
   * `yolo26n.pt`

3. **Start the API Server:**
   ```bash
   uvicorn final3:app --reload --port 8000
   ```

4. **Access the Web Interface:**
   Open [http://localhost:8000/](http://localhost:8000/) in your browser.
