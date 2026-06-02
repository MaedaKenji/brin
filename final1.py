"""
final.py — Unified Surrounding Awareness API Service.

Merges YOLO object/action detection (inference2.py) with GPS POI awareness (gps.py).
Provides both a REST API and a web-based demo interface.

Usage:
    uvicorn final:app --reload --port 8000
    # API:  POST /api/analyze  (multipart form with image/video/webcam frame + optional lat/lng)
    # Web:  GET /
"""

import os
import io
import json
import math
import tempfile
from collections import Counter
from typing import Optional
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from color import detect_dominant_color
from gps import query as gps_query

# ── Constants ────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

# ── Label groups ─────────────────────────────────────────────────────────────────

PERSON_LABELS = {'person', 'person sitting', 'person standing'}

DIRECT_ACTION_LABELS = {'Walking', 'smiling', 'person eating',
                        'person sitting', 'person standing'}

OBJECT_LABELS = {'bicycle', 'bus', 'car', 'chair', 'clothe', 'motorcycle',
                 'shelter', 'sign', 'traffic light', 'truck', 'water bootle',
                 'person', 'door'}

INTERACTION_ACTIONS = {
    'cycling':           {'person': PERSON_LABELS, 'vehicle': {'bicycle'}},
    'riding_motorcycle': {'person': PERSON_LABELS, 'vehicle': {'motorcycle'}},
    'in_car':            {'person': PERSON_LABELS, 'vehicle': {'car'}},
    'in_truck':          {'person': PERSON_LABELS, 'vehicle': {'truck'}},
    'in_bus':            {'person': PERSON_LABELS, 'vehicle': {'bus'}},
    'waiting_bus_stop':  {'person': PERSON_LABELS, 'vehicle': {'sign'}},
}

ACTION_DISPLAY = {
    'cycling':           'cycling',
    'riding_motorcycle': 'riding motorcycle',
    'in_car':            'riding in car',
    'in_truck':          'riding in truck',
    'in_bus':            'riding in bus',
    'waiting_bus_stop':  'waiting at bus stop',
    'Walking':           'walking',
    'smiling':           'smiling',
    'person eating':     'eating',
    'person sitting':    'sitting',
    'person standing':   'standing',
}


# ── Geometry helpers ─────────────────────────────────────────────────────────────

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        inter_area = 0
    else:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0
    return inter_area / union_area


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_distance(box1, box2):
    c1, c2 = center(box1), center(box2)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def boxes_nearby(box1, box2, iou_threshold=0.05, dist_threshold=150):
    if calculate_iou(box1, box2) >= iou_threshold:
        return True
    return center_distance(box1, box2) < dist_threshold


def get_color(cls_id):
    np.random.seed(cls_id)
    return tuple(int(x) for x in np.random.randint(0, 255, 3))


def intersection_area(box1, box2) -> float:
    """Return the area of the intersection rectangle between two boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def box_area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def deduplicate_by_class(
    boxes_info: list[dict],
    iou_threshold: float = 0.45,
) -> list[dict]:
    """
    Per-class Non-Maximum Suppression (NMS).

    When multiple detections of the **same label** overlap with an IoU
    above `iou_threshold`, only the one with the highest confidence is
    kept.  Detections of different classes are never merged.

    Args:
        boxes_info:     List of detection dicts (keys: box, label, conf, cls_id).
        iou_threshold:  Overlap threshold above which a lower-confidence
                        duplicate is suppressed (default 0.45).

    Returns:
        Filtered list with redundant detections removed.
    """
    # Group indices by label
    from collections import defaultdict
    label_groups: dict[str, list[int]] = defaultdict(list)
    for i, b in enumerate(boxes_info):
        label_groups[b['label']].append(i)

    keep = set()
    for label, indices in label_groups.items():
        # Sort by confidence descending within each class
        sorted_idx = sorted(indices, key=lambda i: boxes_info[i]['conf'], reverse=True)
        suppressed = set()
        for i, idx in enumerate(sorted_idx):
            if idx in suppressed:
                continue
            keep.add(idx)
            # Suppress lower-confidence boxes that overlap heavily with this one
            for other_idx in sorted_idx[i + 1:]:
                if other_idx in suppressed:
                    continue
                if calculate_iou(boxes_info[idx]['box'], boxes_info[other_idx]['box']) >= iou_threshold:
                    suppressed.add(other_idx)

    # Preserve original list order
    return [b for i, b in enumerate(boxes_info) if i in keep]


def suppress_contained_doors(
    boxes_info: list[dict],
    containment_threshold: float = 0.80,
) -> list[dict]:
    """
    Door-specific containment filter.

    If a smaller door box is *engulfed* by a larger door box — meaning at
    least `containment_threshold` of the smaller box's area overlaps with
    the larger box — the smaller one is suppressed and only the outer
    (larger) box is kept.

    This is intentionally separate from generic NMS so the containment
    logic is explicit and easy to tune for door detection.

    Args:
        boxes_info:             List of door detection dicts (box, conf, …).
        containment_threshold:  Fraction of the smaller box that must be
                                covered to treat it as contained (default 0.80).

    Returns:
        Filtered list of door detections with contained sub-doors removed.
    """
    n = len(boxes_info)
    suppressed = set()

    for i in range(n):
        if i in suppressed:
            continue
        for j in range(n):
            if i == j or j in suppressed:
                continue
            area_i = box_area(boxes_info[i]['box'])
            area_j = box_area(boxes_info[j]['box'])
            if area_i == 0 or area_j == 0:
                continue
            inter = intersection_area(boxes_info[i]['box'], boxes_info[j]['box'])
            # Determine which box is smaller
            if area_i <= area_j:
                # i is smaller; check if j engulfs i
                if inter / area_i >= containment_threshold:
                    suppressed.add(i)  # suppress the smaller box
                    break              # no need to check more pairs for i
            else:
                # j is smaller; check if i engulfs j
                if inter / area_j >= containment_threshold:
                    suppressed.add(j)

    return [b for i, b in enumerate(boxes_info) if i not in suppressed]


# ── EXIF GPS extraction ─────────────────────────────────────────────────────────

def extract_gps_from_image(image_bytes: bytes) -> Optional[dict]:
    """Try to extract GPS lat/lng from image EXIF metadata."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()
        if exif_data is None:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id in value:
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = value[gps_tag_id]

        if not gps_info:
            return None

        def _to_degrees(value):
            d, m, s = value
            return float(d) + float(m) / 60.0 + float(s) / 3600.0

        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = _to_degrees(gps_info["GPSLatitude"])
            lng = _to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLatitudeRef", "N") == "S":
                lat = -lat
            if gps_info.get("GPSLongitudeRef", "E") == "W":
                lng = -lng
            return {"lat": lat, "lng": lng}
    except Exception:
        pass
    return None


# ── Door direction helper ────────────────────────────────────────────────────────

def get_door_direction(box, image_width: int) -> str:
    """
    Determine simple direction of a door based on its horizontal position.
    Splits the image into thirds: left, front (center), right.
    """
    cx = (box[0] + box[2]) / 2
    third = image_width / 3
    if cx < third:
        return "left"
    elif cx < 2 * third:
        return "front"
    else:
        return "right"


# ── Core analysis function ───────────────────────────────────────────────────────

def analyze_image(
    image_bytes: bytes,
    model: YOLO,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    gps_radius: float = 500,
) -> dict:
    """
    Run full analysis pipeline on a single image.

    1. YOLO detection → objects, actions, colors
    2. Door direction detection
    3. GPS resolution (user input → EXIF fallback → None)
    4. POI lookup if GPS available
    """

    # ── Resolve GPS ──────────────────────────────────────────────────────────
    gps_source = None
    if lat is not None and lng is not None:
        gps_source = "user_input"
    else:
        exif_gps = extract_gps_from_image(image_bytes)
        if exif_gps:
            lat, lng = exif_gps["lat"], exif_gps["lng"]
            gps_source = "exif_metadata"

    # ── Decode image ─────────────────────────────────────────────────────────
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    img_h, img_w = img.shape[:2]

    # ── YOLO inference ───────────────────────────────────────────────────────
    results = model.predict(source=img, conf=0.1, iou=0.1, agnostic_nms=False)
    result = results[0]

    boxes_info = []
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        cls_id = int(cls)
        label = result.names[cls_id]
        boxes_info.append({
            'box':    box.tolist(),
            'cls_id': cls_id,
            'label':  label,
            'conf':   float(conf),
        })

    # ── Deduplicate overlapping detections of the same class ─────────────────
    # Suppresses lower-confidence boxes that heavily overlap (IoU ≥ 0.45) with
    # a higher-confidence detection of the same label.
    boxes_info = deduplicate_by_class(boxes_info, iou_threshold=0.45)

    # ── Index sets ───────────────────────────────────────────────────────────
    person_indices     = [i for i, b in enumerate(boxes_info) if b['label'] in PERSON_LABELS]
    bicycle_indices    = [i for i, b in enumerate(boxes_info) if b['label'] == 'bicycle']
    motorcycle_indices = [i for i, b in enumerate(boxes_info) if b['label'] == 'motorcycle']
    car_indices        = [i for i, b in enumerate(boxes_info) if b['label'] == 'car']
    truck_indices      = [i for i, b in enumerate(boxes_info) if b['label'] == 'truck']
    bus_indices        = [i for i, b in enumerate(boxes_info) if b['label'] == 'bus']
    sign_indices       = [i for i, b in enumerate(boxes_info) if b['label'] == 'sign']

    # ── Interaction actions ──────────────────────────────────────────────────
    def count_interactions(p_indices, v_indices):
        count = 0
        for p_idx in p_indices:
            for v_idx in v_indices:
                if boxes_nearby(boxes_info[p_idx]['box'], boxes_info[v_idx]['box']):
                    count += 1
                    break
        return count

    interaction_map = {
        'cycling':           count_interactions(person_indices, bicycle_indices),
        'riding_motorcycle': count_interactions(person_indices, motorcycle_indices),
        'in_car':            count_interactions(person_indices, car_indices),
        'in_truck':          count_interactions(person_indices, truck_indices),
        'in_bus':            count_interactions(person_indices, bus_indices),
        'waiting_bus_stop':  count_interactions(person_indices, sign_indices),
    }

    # Direct action counts
    direct_action_counts = {}
    for action_label in DIRECT_ACTION_LABELS:
        cnt = sum(1 for b in boxes_info if b['label'] == action_label)
        if cnt > 0:
            display = ACTION_DISPLAY.get(action_label, action_label)
            direct_action_counts[display] = cnt

    # ── Object entries with color ────────────────────────────────────────────
    object_entries = {}
    for i, b in enumerate(boxes_info):
        label = b['label']
        if label in DIRECT_ACTION_LABELS:
            continue
        colors = detect_dominant_color(img, b['box'], n_colors=3)
        entry = {
            "id":         i,
            "confidence": round(b['conf'], 3),
            "colors":     colors,
        }
        object_entries.setdefault(label, []).append(entry)

    # ── Build objects section ────────────────────────────────────────────────
    objects_section = {}
    for label, instances in object_entries.items():
        objects_section[label] = {
            "count":     len(instances),
            "instances": instances,
        }

    # ── Build actions section ────────────────────────────────────────────────
    actions_section = {}
    for key, cnt in interaction_map.items():
        if cnt > 0:
            actions_section[ACTION_DISPLAY[key]] = {"count": cnt}
    for display, cnt in direct_action_counts.items():
        actions_section[display] = {"count": cnt}

    # ── Door direction detection ─────────────────────────────────────────────
    # First, filter out door boxes that are engulfed by a larger door box so
    # that only the outermost (real) door is counted per physical door.
    raw_door_boxes = [b for b in boxes_info if b['label'] == 'door']
    filtered_doors = suppress_contained_doors(raw_door_boxes, containment_threshold=0.80)

    door_info = []
    for i, b in enumerate(filtered_doors):
        direction = get_door_direction(b['box'], img_w)
        door_info.append({
            "id":          i,
            "confidence":  round(b['conf'], 3),
            "direction":   direction,
            "description": f"There is a door on your {direction}",
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_section = {}
    for label, data in objects_section.items():
        summary_section[label] = data["count"]
    for action, data in actions_section.items():
        summary_section[action] = data["count"]

    # ── GPS / POI section ────────────────────────────────────────────────────
    gps_section = None
    if lat is not None and lng is not None:
        poi_result = gps_query(lat, lng, gps_radius)
        gps_section = {
            "source":     gps_source,
            "lat":        lat,
            "lng":        lng,
            "poi_result": poi_result,
        }

    # ── Assemble output ──────────────────────────────────────────────────────
    output = {
        "objects":  objects_section,
        "actions":  actions_section,
        "doors":    door_info if door_info else None,
        "summary":  summary_section,
        "gps":      gps_section,
    }
    return output


def _merge_video_frame_results(frame_results: list[dict]) -> dict:
    """
    Build a video-level summary from sampled frame analyses.

    Counts are reported as the maximum count seen in any sampled frame. This
    avoids treating the same object as many separate objects just because it
    appears across multiple frames.
    """
    objects: dict[str, dict] = {}
    actions: dict[str, dict] = {}
    summary: dict[str, int] = {}
    door_direction_counts: Counter = Counter()

    for frame in frame_results:
        result = frame["analysis"]

        for label, data in (result.get("objects") or {}).items():
            entry = objects.setdefault(label, {
                "count": 0,
                "detections_in_frames": 0,
                "instances": [],
            })
            entry["count"] = max(entry["count"], data.get("count", 0))
            entry["detections_in_frames"] += 1
            for instance in data.get("instances", []):
                copied = dict(instance)
                copied["frame_index"] = frame["frame_index"]
                copied["timestamp_sec"] = frame["timestamp_sec"]
                entry["instances"].append(copied)

        for label, data in (result.get("actions") or {}).items():
            entry = actions.setdefault(label, {"count": 0, "detections_in_frames": 0})
            entry["count"] = max(entry["count"], data.get("count", 0))
            entry["detections_in_frames"] += 1

        for door in (result.get("doors") or []):
            door_direction_counts[door["direction"]] += 1

    doors = [
        {
            "direction": direction,
            "detections_in_frames": count,
            "description": f"There is a door on your {direction}",
        }
        for direction, count in door_direction_counts.items()
    ]

    for label, data in objects.items():
        summary[label] = data["count"]
    for label, data in actions.items():
        summary[label] = data["count"]

    return {
        "objects": objects,
        "actions": actions,
        "doors": doors if doors else None,
        "summary": summary,
    }


def analyze_video(
    video_bytes: bytes,
    model: YOLO,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    gps_radius: float = 500,
    sample_interval_sec: float = 1.0,
    max_frames: int = 30,
    suffix: str = ".mp4",
) -> dict:
    """
    Run the image analysis pipeline over sampled video frames.

    The uploaded bytes are written to a temporary file because OpenCV's
    VideoCapture expects a filesystem path for common container formats.
    """
    if sample_interval_sec <= 0:
        raise ValueError("sample_interval_sec must be greater than 0")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than 0")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not decode video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (total_frames / fps) if fps > 0 and total_frames > 0 else None
        frame_step = max(1, int(round((fps or 1) * sample_interval_sec)))

        frame_results = []
        frame_index = 0
        while len(frame_results) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frame_step == 0:
                encoded, buffer = cv2.imencode(".jpg", frame)
                if encoded:
                    analysis = analyze_image(
                        image_bytes=buffer.tobytes(),
                        model=model,
                        lat=None,
                        lng=None,
                        gps_radius=gps_radius,
                    )
                    frame_results.append({
                        "frame_index": frame_index,
                        "timestamp_sec": round(frame_index / fps, 3) if fps > 0 else None,
                        "analysis": analysis,
                    })
            frame_index += 1

        cap.release()

        if not frame_results:
            raise ValueError("No readable frames found in video")

        aggregate = _merge_video_frame_results(frame_results)
        gps_section = None
        if lat is not None and lng is not None:
            gps_section = {
                "source": "user_input",
                "lat": lat,
                "lng": lng,
                "poi_result": gps_query(lat, lng, gps_radius),
            }

        return {
            "media_type": "video",
            "video": {
                "fps": fps,
                "total_frames": total_frames,
                "duration_sec": duration_sec,
                "sample_interval_sec": sample_interval_sec,
                "sampled_frames": len(frame_results),
                "max_frames": max_frames,
            },
            "objects": aggregate["objects"],
            "actions": aggregate["actions"],
            "doors": aggregate["doors"],
            "summary": aggregate["summary"],
            "gps": gps_section,
            "frames": frame_results,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── FastAPI app ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Surrounding Awareness API",
    description="YOLO object/action detection + GPS POI awareness in one service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


@app.on_event("startup")
def startup():
    _get_model()


# ── API endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def api_analyze(
    image: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    gps_radius: Optional[float] = Form(500),
    sample_interval_sec: Optional[float] = Form(1.0),
    max_frames: Optional[int] = Form(30),
):
    """
    Analyze an uploaded image or video.

    - **image**: Image or video file. The field name stays `image` for
      backward compatibility.
    - **lat**: Optional latitude (decimal degrees). If not given, tries EXIF.
    - **lng**: Optional longitude (decimal degrees). If not given, tries EXIF.
    - **gps_radius**: POI search radius in meters (default 500)
    - **sample_interval_sec**: Video frame sampling interval in seconds
    - **max_frames**: Maximum sampled video frames to analyze
    """
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload file")

    content_type = (image.content_type or "").lower()
    filename = (image.filename or "").lower()
    video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
    video_suffix = os.path.splitext(filename)[1] if filename else ".mp4"
    is_video = content_type.startswith("video/") or filename.endswith(video_extensions)

    try:
        if is_video:
            result = analyze_video(
                video_bytes=contents,
                model=_get_model(),
                lat=lat,
                lng=lng,
                gps_radius=gps_radius or 500,
                sample_interval_sec=sample_interval_sec or 1.0,
                max_frames=max_frames or 30,
                suffix=video_suffix if video_suffix in video_extensions else ".mp4",
            )
        else:
            result = analyze_image(
                image_bytes=contents,
                model=_get_model(),
                lat=lat,
                lng=lng,
                gps_radius=gps_radius or 500,
            )
            result["media_type"] = "image"
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content=result)


# ── Web demo ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def web_demo():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Surrounding Awareness — Demo</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0f0f1a;--surface:#1a1a2e;--surface2:#25254a;--border:#33336a;
  --text:#e8e8f0;--text2:#9999bb;--accent:#6c63ff;--accent2:#ff6584;
  --green:#00d68f;--radius:14px;
}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);
  min-height:100vh;display:flex;flex-direction:column;align-items:center;
  padding:2rem 1rem}
h1{font-size:1.8rem;font-weight:700;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin-bottom:.4rem}
.subtitle{color:var(--text2);font-size:.9rem;margin-bottom:2rem}
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:1.5rem;width:100%;max-width:720px;
  margin-bottom:1.5rem;transition:box-shadow .3s}
.card:hover{box-shadow:0 0 30px rgba(108,99,255,.15)}
.card h2{font-size:1rem;font-weight:600;margin-bottom:1rem;color:var(--accent)}
label{display:block;font-size:.85rem;font-weight:500;color:var(--text2);margin-bottom:.35rem}
input[type=file]{display:none}
.file-label{display:flex;align-items:center;justify-content:center;gap:.5rem;
  border:2px dashed var(--border);border-radius:var(--radius);padding:2rem;
  cursor:pointer;transition:border-color .3s,background .3s;font-size:.9rem;color:var(--text2)}
.file-label:hover{border-color:var(--accent);background:rgba(108,99,255,.06)}
.file-label.has-file{border-color:var(--green);color:var(--green)}
.row{display:flex;gap:1rem;margin-top:1rem}
.row .field{flex:1}
.row input[type=number]{width:100%;padding:.6rem .8rem;border-radius:8px;
  border:1px solid var(--border);background:var(--surface2);color:var(--text);
  font-size:.9rem;outline:none;transition:border-color .3s}
.row input[type=number]:focus{border-color:var(--accent)}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:.5rem;
  margin-top:1.2rem;padding:.75rem 2rem;border:none;border-radius:10px;
  background:linear-gradient(135deg,var(--accent),#8b5cf6);color:#fff;
  font-size:.95rem;font-weight:600;cursor:pointer;transition:transform .15s,box-shadow .3s;
  width:100%}
.btn:hover{transform:translateY(-2px);box-shadow:0 6px 24px rgba(108,99,255,.35)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.btn.secondary{background:var(--surface2);border:1px solid var(--border);color:var(--text)}
.btn.stop{background:linear-gradient(135deg,var(--accent2),#ef4444)}
.camera-actions{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.spinner{width:18px;height:18px;border:2px solid rgba(255,255,255,.3);
  border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}
#preview,#video-preview,#camera-preview{max-width:100%;border-radius:10px;margin-top:1rem;display:none;
  border:1px solid var(--border)}
#camera-canvas{display:none}
.status{margin-top:.8rem;color:var(--text2);font-size:.85rem;min-height:1.2rem}
.video-options{display:none}
#result-box{display:none}
#result{background:var(--surface2);border-radius:10px;padding:1rem;
  overflow-x:auto;font-size:.82rem;line-height:1.5;white-space:pre-wrap;
  max-height:600px;overflow-y:auto;color:#c8c8e0;border:1px solid var(--border)}
.tag{display:inline-block;padding:.2rem .6rem;border-radius:6px;font-size:.75rem;
  font-weight:600;margin-right:.4rem;margin-bottom:.3rem}
.tag-obj{background:rgba(108,99,255,.15);color:var(--accent)}
.tag-act{background:rgba(255,101,132,.15);color:var(--accent2)}
.tag-door{background:rgba(0,214,143,.15);color:var(--green)}
.tag-gps{background:rgba(255,165,0,.15);color:#ffa500}
#quick-summary{margin-bottom:1rem}
</style>
</head>
<body>
<h1>🔍 Surrounding Awareness</h1>
<p class="subtitle">Upload media or stream your webcam for continuous surrounding awareness</p>

<div class="card">
  <h2>📤 Upload & Settings</h2>
  <label class="file-label" id="file-label" for="image-input">
    <span id="file-text">Click or drag to upload an image or video</span>
  </label>
  <input type="file" id="image-input" accept="image/*,video/*">
  <img id="preview" alt="preview">
  <video id="video-preview" controls muted playsinline></video>
  <div class="row">
    <div class="field">
      <label for="lat-input">Latitude (optional)</label>
      <input type="number" id="lat-input" step="any" placeholder="e.g. 24.968">
    </div>
    <div class="field">
      <label for="lng-input">Longitude (optional)</label>
      <input type="number" id="lng-input" step="any" placeholder="e.g. 121.191">
    </div>
    <div class="field">
      <label for="radius-input">Radius (m)</label>
      <input type="number" id="radius-input" value="500" min="1" max="5000">
    </div>
  </div>
  <div class="row video-options" id="video-options">
    <div class="field">
      <label for="sample-input">Sample every (sec)</label>
      <input type="number" id="sample-input" value="1" min="0.1" step="0.1">
    </div>
    <div class="field">
      <label for="max-frames-input">Max frames</label>
      <input type="number" id="max-frames-input" value="30" min="1" max="300">
    </div>
  </div>
  <button class="btn" id="analyze-btn" disabled>
    <span class="spinner" id="spinner"></span>
    <span id="btn-text">Analyze Media</span>
  </button>
</div>

<div class="card">
  <h2>Live Webcam</h2>
  <video id="camera-preview" autoplay muted playsinline></video>
  <canvas id="camera-canvas"></canvas>
  <div class="row">
    <div class="field">
      <label for="camera-interval-input">Infer every (sec)</label>
      <input type="number" id="camera-interval-input" value="1" min="0.2" step="0.1">
    </div>
  </div>
  <div class="camera-actions">
    <button class="btn secondary" id="start-camera-btn">Start Webcam</button>
    <button class="btn stop" id="stop-camera-btn" disabled>Stop Webcam</button>
  </div>
  <div class="status" id="camera-status"></div>
</div>

<div class="card" id="result-box">
  <h2>📊 Results</h2>
  <div id="quick-summary"></div>
  <pre id="result"></pre>
</div>

<script>
const fileInput=document.getElementById('image-input'),
      fileLabel=document.getElementById('file-label'),
      fileText=document.getElementById('file-text'),
      preview=document.getElementById('preview'),
      videoPreview=document.getElementById('video-preview'),
      videoOptions=document.getElementById('video-options'),
      cameraPreview=document.getElementById('camera-preview'),
      cameraCanvas=document.getElementById('camera-canvas'),
      startCameraBtn=document.getElementById('start-camera-btn'),
      stopCameraBtn=document.getElementById('stop-camera-btn'),
      cameraIntervalInput=document.getElementById('camera-interval-input'),
      cameraStatus=document.getElementById('camera-status'),
      btn=document.getElementById('analyze-btn'),
      btnText=document.getElementById('btn-text'),
      spinner=document.getElementById('spinner'),
      resultBox=document.getElementById('result-box'),
      resultPre=document.getElementById('result'),
      quickSummary=document.getElementById('quick-summary');

let cameraStream=null,
    cameraTimer=null,
    cameraRequestInFlight=false,
    cameraFrameNumber=0;

function renderResult(data){
  let tags='';
  if(data.objects){
    for(const[k,v]of Object.entries(data.objects))
      tags+=`<span class="tag tag-obj">${k}: ${v.count}</span>`;
  }
  if(data.actions){
    for(const[k,v]of Object.entries(data.actions))
      tags+=`<span class="tag tag-act">${k}: ${v.count}</span>`;
  }
  if(data.doors&&data.doors.length){
    for(const d of data.doors)
      tags+=`<span class="tag tag-door">${d.description}</span>`;
  }
  if(data.gps){
    tags+=`<span class="tag tag-gps">GPS: ${data.gps.source}</span>`;
    if(data.gps.poi_result&&data.gps.poi_result.nearest)
      tags+=`<span class="tag tag-gps">Nearest: ${data.gps.poi_result.nearest.name}</span>`;
  }
  quickSummary.innerHTML=tags||'<span class="tag tag-obj">No detections</span>';
  resultPre.textContent=JSON.stringify(data,null,2);
  resultBox.style.display='block';
}

function appendLocationFields(fd){
  const lat=document.getElementById('lat-input').value;
  const lng=document.getElementById('lng-input').value;
  const radius=document.getElementById('radius-input').value;
  if(lat)fd.append('lat',lat);
  if(lng)fd.append('lng',lng);
  if(radius)fd.append('gps_radius',radius);
}

fileInput.addEventListener('change',()=>{
  if(fileInput.files.length){
    const f=fileInput.files[0];
    fileText.textContent=f.name;
    fileLabel.classList.add('has-file');
    const url=URL.createObjectURL(f);
    const isVideo=f.type.startsWith('video/');
    preview.style.display='none';
    videoPreview.style.display='none';
    if(isVideo){
      videoPreview.src=url;
      videoPreview.style.display='block';
      videoOptions.style.display='flex';
      btnText.textContent='Analyze Video';
    }else{
      preview.src=url;
      preview.style.display='block';
      videoOptions.style.display='none';
      btnText.textContent='Analyze Image';
    }
    btn.disabled=false;
  }
});

btn.addEventListener('click',async()=>{
  if(!fileInput.files.length)return;
  stopCamera();
  btn.disabled=true;btnText.textContent='Analyzing...';spinner.style.display='inline-block';
  resultBox.style.display='none';

  const fd=new FormData();
  fd.append('image',fileInput.files[0]);
  const sampleEvery=document.getElementById('sample-input').value;
  const maxFrames=document.getElementById('max-frames-input').value;
  appendLocationFields(fd);
  if(fileInput.files[0].type.startsWith('video/')){
    if(sampleEvery)fd.append('sample_interval_sec',sampleEvery);
    if(maxFrames)fd.append('max_frames',maxFrames);
  }

  try{
    const res=await fetch('/api/analyze',{method:'POST',body:fd});
    const data=await res.json();
    renderResult(data);
  }catch(e){
    resultPre.textContent='Error: '+e.message;
    quickSummary.innerHTML='';
    resultBox.style.display='block';
  }finally{
    btn.disabled=false;
    btnText.textContent=fileInput.files[0]?.type.startsWith('video/')?'Analyze Video':'Analyze Image';
    spinner.style.display='none';
  }
});

async function analyzeCameraFrame(){
  if(!cameraStream||cameraRequestInFlight||cameraPreview.videoWidth===0)return;

  cameraRequestInFlight=true;
  cameraFrameNumber+=1;
  cameraStatus.textContent=`Inferencing frame ${cameraFrameNumber}...`;

  cameraCanvas.width=cameraPreview.videoWidth;
  cameraCanvas.height=cameraPreview.videoHeight;
  const ctx=cameraCanvas.getContext('2d');
  ctx.drawImage(cameraPreview,0,0,cameraCanvas.width,cameraCanvas.height);

  try{
    const blob=await new Promise(resolve=>cameraCanvas.toBlob(resolve,'image/jpeg',0.85));
    if(!blob)throw new Error('Could not capture webcam frame');

    const fd=new FormData();
    fd.append('image',blob,`webcam-frame-${cameraFrameNumber}.jpg`);
    appendLocationFields(fd);

    const res=await fetch('/api/analyze',{method:'POST',body:fd});
    const data=await res.json();
    if(!res.ok)throw new Error(data.detail||'Webcam inference failed');
    data.media_type='webcam';
    data.webcam={frame_number:cameraFrameNumber,captured_at:new Date().toISOString()};
    renderResult(data);
    cameraStatus.textContent=`Live inferencing. Last frame: ${cameraFrameNumber}`;
  }catch(e){
    cameraStatus.textContent='Webcam error: '+e.message;
  }finally{
    cameraRequestInFlight=false;
  }
}

async function startCamera(){
  try{
    if(cameraStream)return;
    resultBox.style.display='none';
    cameraFrameNumber=0;
    cameraStream=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:'environment'},
      audio:false,
    });
    cameraPreview.srcObject=cameraStream;
    cameraPreview.style.display='block';
    startCameraBtn.disabled=true;
    stopCameraBtn.disabled=false;
    cameraStatus.textContent='Camera started. Waiting for first frame...';

    const intervalMs=Math.max(200,Number(cameraIntervalInput.value||1)*1000);
    cameraTimer=setInterval(analyzeCameraFrame,intervalMs);
    cameraPreview.onloadedmetadata=()=>analyzeCameraFrame();
  }catch(e){
    cameraStatus.textContent='Could not start webcam: '+e.message;
    stopCamera();
  }
}

function stopCamera(){
  if(cameraTimer){
    clearInterval(cameraTimer);
    cameraTimer=null;
  }
  if(cameraStream){
    for(const track of cameraStream.getTracks())track.stop();
    cameraStream=null;
  }
  cameraRequestInFlight=false;
  cameraPreview.srcObject=null;
  cameraPreview.style.display='none';
  startCameraBtn.disabled=false;
  stopCameraBtn.disabled=true;
  if(cameraStatus)cameraStatus.textContent=cameraFrameNumber?'Camera stopped.':'';
}

startCameraBtn.addEventListener('click',startCamera);
stopCameraBtn.addEventListener('click',stopCamera);
</script>
</body>
</html>"""


# ── CLI entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting Surrounding Awareness API on http://localhost:8000")
    print("  API:  POST /api/analyze")
    print("  Demo: GET  /")
    uvicorn.run(app, host="0.0.0.0", port=8000)
