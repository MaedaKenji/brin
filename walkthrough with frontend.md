# Walkthrough: Unified Surrounding Awareness API Service

This walkthrough details the architecture, features, and endpoints of [final3.py](file:///d:/Code/Python/brin/final3.py), which serves as a unified backend service merging computer vision (YOLO) and GPS/altitude spatial context.

## Key Features

1. **Unified API Service**: Supports analysis of both static images and video files.
2. **Dynamic YOLO Model Routing**:
   - **`auto` (default)**: Dynamically selects between `indoor.pt` and `outdoor.pt` depending on GPS distance to known buildings in [poi_buildings_english.csv](file:///d:/Code/Python/brin/poi_buildings_english.csv).
   - **`indoor` / `outdoor` / `base`**: Explicit overrides to bypass automatic selection (`base` uses `yolo26n.pt`).
3. **Advanced Detection Post-Processing**:
   - **Per-Class NMS (Non-Maximum Suppression)**: Overlapping detections of the same label above a threshold (IoU $\ge$ 0.45) are deduplicated.
   - **Door-Specific Containment**: Smaller doors engulfed by larger doors are suppressed to avoid double-counting.
   - **Door Direction estimation**: Identifies if a door is on the "left", "front", or "right" of the image frame.
4. **Contextual Metadata Merging**:
   - **GPS Coordinates**: Decodes GPS coordinate parameters, falls back to EXIF tags from JPEG images, and matches them to physical Points of Interest (POIs).
   - **Floor/Altitude Estimation**: Translates GPS altitude into estimated floor numbers using standard altitude bands.

---

## Code Architecture

### 1. Data Models and Constants
- `ModelMode` [final3.py:L65-71](file:///d:/Code/Python/brin/final3.py#L65-L71): Enum defining `auto`, `indoor`, `outdoor`, and `base` options.
- Label groupings: Differentiates direct action labels (e.g. "Walking"), people labels, vehicles, and objects.

### 2. Geometry Helpers
- `calculate_iou` [final3.py:L129](file:///d:/Code/Python/brin/final3.py#L129): Calculates Intersection over Union of two bounding boxes.
- `deduplicate_by_class` [final3.py:L184](file:///d:/Code/Python/brin/final3.py#L184): Implements per-class NMS.
- `suppress_contained_doors` [final3.py:L233](file:///d:/Code/Python/brin/final3.py#L233): Filters engulfed door detections.

### 3. Location and Environment Logic
- `determine_environment` [final3.py:L781](file:///d:/Code/Python/brin/final3.py#L781): Checks distance to nearest POI in `poi_buildings_english.csv` using the Haversine formula. If the user is within `indoor_radius_m` (default: 30m) of a building, environment is classified as `"indoor"`.
- `resolve_model` [final3.py:L827](file:///d:/Code/Python/brin/final3.py#L827): Resolves the appropriate `.pt` model file path and creates/returns the YOLO object.

### 4. Video Sampling and Aggregation
- `analyze_video` [final3.py:L622](file:///d:/Code/Python/brin/final3.py#L622): Writes incoming video bytes to a temporary file, samples frames at `sample_interval_sec` (up to `max_frames`), runs `analyze_image` on each frame, and merges results using `_merge_video_frame_results`.

---

## API Endpoints

### 1. Health Check
* **GET `/api/health`**
  * Returns `{"status": "ok"}`

### 2. Surrounding Awareness Analysis
* **POST `/api/analyze`**
  * **Payload (Multipart Form)**:
    * `image` (File, required): An image or video file.
    * `lat` / `lng` (float, optional): Latitude and longitude.
    * `altitude` (float, optional): Altitude in meters for floor estimate.
    * `model` (string, optional): `"auto"` | `"indoor"` | `"outdoor"` | `"base"`.
    * `indoor_radius_m` (float, optional): Distance threshold in meters.
  * **Returns**: A JSON payload containing:
    * `objects`: Detected items, counts, confidence, and dominant colors.
    * `actions`: Interaction states (e.g. sitting, walking, riding in car) and counts.
    * `doors`: Left/front/right directions and description strings.
    * `gps`: Distance to nearest POIs, resolved coordinate source.
    * `floor`: Estimated floor level.
    * `model_info`: Detail on which YOLO model was selected and why.

---

## Requirements and Prerequisites

Before running the application, make sure you have the following installed and configured:

1. **Python 3.9+**: Recommended version is Python 3.10 or 3.11.
2. **Model Files**: Ensure the following models exist in the root folder of the project:
   - `indoor.pt` (Indoor YOLO model)
   - `outdoor.pt` (Outdoor YOLO model)
   - `yolo26n.pt` (Base YOLO model)
3. **Data Files**: 
   - `poi_buildings_english.csv` (used for calculating POI and indoor/outdoor proximity)
4. **Python Dependencies**: Listed in [requirements.txt](file:///d:/Code/Python/brin/requirements.txt):
   - `fastapi` and `uvicorn` (to run the web API)
   - `python-multipart` (to parse uploaded images and files)
   - `ultralytics` (for YOLO predictions)
   - `opencv-python` and `pillow` (for image processing and metadata extraction)
   - `numpy` and `scipy` (for geometry calculations and spatial indexing trees)

---

## Detailed Step-by-Step Run Guide

For users who are new to running Python web services, follow these step-by-step instructions:

### Step 1: Open a Terminal or Command Prompt
Open your terminal (PowerShell / Command Prompt on Windows, or Terminal on macOS/Linux) and navigate to the project directory:
```powershell
cd d:\Code\Python\brin
```

### Step 2: Create a Virtual Environment (Recommended)
A virtual environment keeps the project dependencies isolated from your global system installations.
* **On Windows**:
  ```powershell
  python -m venv venv
  ```
* **On macOS/Linux**:
  ```bash
  python3 -m venv venv
  ```

### Step 3: Activate the Virtual Environment
Activate the environment so that any packages you install go directly into this workspace.
* **On Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **On Windows (Command Prompt)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```
* **On macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install Dependencies
Install all required libraries using the package manager (`pip`):
```bash
pip install -r requirements.txt
```

### Step 5: Start the API Server
Start the development server using `uvicorn`:
```bash
uvicorn final3:app --reload --port 8000
```

Once started, you will see output in the terminal indicating the server is running:
```text
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Testing and Using the App

### 1. Web Demo Interface
Open your web browser and go to:
[http://localhost:8000/](http://localhost:8000/)

This will load the interactive frontend dashboard, allowing you to upload images or video and see real-time detections and GPS mappings.

### 2. Direct API Endpoint testing (cURL)
You can send structured requests directly to the API endpoint `POST /api/analyze` using CLI tools like `curl`:
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "image=@/path/to/your/test_image.jpg" \
  -F "lat=24.968" \
  -F "lng=121.191" \
  -F "model=auto"
```

