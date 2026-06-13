"""
final3.py — Unified Surrounding Awareness API Service.

Merges YOLO object/action detection with GPS POI awareness.
Provides both a REST API and a web-based demo interface.

New in this version:
  - Automatic model switching: indoor.pt / outdoor.pt selected by GPS proximity
    to buildings in poi_buildings_english.csv.
  - Explicit model override via `model` form field (auto|indoor|outdoor|base).
  - Base YOLO model option via `model=base` (uses yolo26n.pt).

Usage:
    uvicorn final3:app --reload --port 8000
    # API:  POST /api/analyze  (multipart form with image/video/webcam frame + optional lat/lng)
    # Web:  GET /
"""

import csv
import os
import io
import json
import math
import tempfile
from collections import Counter
from enum import Enum
from typing import Optional

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from color import detect_dominant_color
from gps import query as gps_query

# ── Constants ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model file paths
MODEL_INDOOR_PATH = os.path.join(BASE_DIR, "indoor.pt")
MODEL_OUTDOOR_PATH = os.path.join(BASE_DIR, "outdoor.pt")
MODEL_BASE_PATH = os.path.join(BASE_DIR, "yolo26n.pt")

# POI CSV used for indoor/outdoor decision
POI_CSV_PATH = os.path.join(BASE_DIR, "poi_buildings_english.csv")

# Default distance (metres) from a building centroid to consider the user indoors
DEFAULT_INDOOR_RADIUS_M = 30.0

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
FLOOR_BASE_ALTITUDE_M = 100.0
FLOOR_HEIGHT_M = 5.0
MAX_FLOOR = 15

# ── Model mode ────────────────────────────────────────────────────────────────────


class ModelMode(str, Enum):
    """Controls which YOLO model is used for inference."""

    AUTO = "auto"  # GPS-based automatic selection (default)
    INDOOR = "indoor"  # Always use indoor.pt
    OUTDOOR = "outdoor"  # Always use outdoor.pt
    BASE = "base"  # Always use yolo26n.pt (base model)


# ── Label groups ─────────────────────────────────────────────────────────────────

PERSON_LABELS = {"person", "person sitting", "person standing"}

DIRECT_ACTION_LABELS = {
    "Walking",
    "smiling",
    "person eating",
    "person sitting",
    "person standing",
}

OBJECT_LABELS = {
    "bicycle",
    "bus",
    "car",
    "chair",
    "clothe",
    "motorcycle",
    "shelter",
    "sign",
    "traffic light",
    "truck",
    "water bootle",
    "person",
    "door",
}

INTERACTION_ACTIONS = {
    "cycling": {"person": PERSON_LABELS, "vehicle": {"bicycle"}},
    "riding_motorcycle": {"person": PERSON_LABELS, "vehicle": {"motorcycle"}},
    "in_car": {"person": PERSON_LABELS, "vehicle": {"car"}},
    "in_truck": {"person": PERSON_LABELS, "vehicle": {"truck"}},
    "in_bus": {"person": PERSON_LABELS, "vehicle": {"bus"}},
    "waiting_bus_stop": {"person": PERSON_LABELS, "vehicle": {"sign"}},
}

ACTION_DISPLAY = {
    "cycling": "cycling",
    "riding_motorcycle": "riding motorcycle",
    "in_car": "riding in car",
    "in_truck": "riding in truck",
    "in_bus": "riding in bus",
    "waiting_bus_stop": "waiting at bus stop",
    "Walking": "walking",
    "smiling": "smiling",
    "person eating": "eating",
    "person sitting": "sitting",
    "person standing": "standing",
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
        label_groups[b["label"]].append(i)

    keep = set()
    for label, indices in label_groups.items():
        # Sort by confidence descending within each class
        sorted_idx = sorted(indices, key=lambda i: boxes_info[i]["conf"], reverse=True)
        suppressed = set()
        for i, idx in enumerate(sorted_idx):
            if idx in suppressed:
                continue
            keep.add(idx)
            # Suppress lower-confidence boxes that overlap heavily with this one
            for other_idx in sorted_idx[i + 1 :]:
                if other_idx in suppressed:
                    continue
                if (
                    calculate_iou(boxes_info[idx]["box"], boxes_info[other_idx]["box"])
                    >= iou_threshold
                ):
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
            area_i = box_area(boxes_info[i]["box"])
            area_j = box_area(boxes_info[j]["box"])
            if area_i == 0 or area_j == 0:
                continue
            inter = intersection_area(boxes_info[i]["box"], boxes_info[j]["box"])
            # Determine which box is smaller
            if area_i <= area_j:
                # i is smaller; check if j engulfs i
                if inter / area_i >= containment_threshold:
                    suppressed.add(i)  # suppress the smaller box
                    break  # no need to check more pairs for i
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


def estimate_floor_from_altitude(altitude_m: Optional[float]) -> Optional[dict]:
    """
    Temporary floor estimator using fixed altitude bands above sea level.

    The placeholder rule is:
    floor 1 = 100-105 m, floor 2 = 105-110 m, and so on up to floor 15.
    This can be replaced with survey data later without touching the API
    contract.
    """
    if altitude_m is None:
        return None

    top_altitude = FLOOR_BASE_ALTITUDE_M + (FLOOR_HEIGHT_M * MAX_FLOOR)
    if altitude_m < FLOOR_BASE_ALTITUDE_M or altitude_m >= top_altitude:
        return None

    floor_number = int((altitude_m - FLOOR_BASE_ALTITUDE_M) // FLOOR_HEIGHT_M) + 1
    band_min = FLOOR_BASE_ALTITUDE_M + ((floor_number - 1) * FLOOR_HEIGHT_M)
    band_max = band_min + FLOOR_HEIGHT_M

    return {
        "source": "user_input",
        "altitude_m": altitude_m,
        "floor": floor_number,
        "range_m": {
            "min": band_min,
            "max": band_max,
        },
        "description": f"Estimated floor {floor_number} from altitude {altitude_m:g} m",
    }


# ── Core analysis function ───────────────────────────────────────────────────────


def analyze_image(
    image_bytes: bytes,
    model: YOLO,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    altitude: Optional[float] = None,
    gps_radius: float = 500,
) -> dict:
    """
    Run full analysis pipeline on a single image.

    1. YOLO detection → objects, actions, colors
    2. Door direction detection
    3. GPS resolution (user input → EXIF fallback → None)
    4. POI lookup if GPS available
    5. Floor estimate from optional altitude above sea level
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
        boxes_info.append(
            {
                "box": box.tolist(),
                "cls_id": cls_id,
                "label": label,
                "conf": float(conf),
            }
        )

    # ── Deduplicate overlapping detections of the same class ─────────────────
    # Suppresses lower-confidence boxes that heavily overlap (IoU ≥ 0.45) with
    # a higher-confidence detection of the same label.
    boxes_info = deduplicate_by_class(boxes_info, iou_threshold=0.45)

    # ── Index sets ───────────────────────────────────────────────────────────
    person_indices = [
        i for i, b in enumerate(boxes_info) if b["label"] in PERSON_LABELS
    ]
    bicycle_indices = [i for i, b in enumerate(boxes_info) if b["label"] == "bicycle"]
    motorcycle_indices = [
        i for i, b in enumerate(boxes_info) if b["label"] == "motorcycle"
    ]
    car_indices = [i for i, b in enumerate(boxes_info) if b["label"] == "car"]
    truck_indices = [i for i, b in enumerate(boxes_info) if b["label"] == "truck"]
    bus_indices = [i for i, b in enumerate(boxes_info) if b["label"] == "bus"]
    sign_indices = [i for i, b in enumerate(boxes_info) if b["label"] == "sign"]

    # ── Interaction actions ──────────────────────────────────────────────────
    def count_interactions(p_indices, v_indices):
        count = 0
        for p_idx in p_indices:
            for v_idx in v_indices:
                if boxes_nearby(boxes_info[p_idx]["box"], boxes_info[v_idx]["box"]):
                    count += 1
                    break
        return count

    interaction_map = {
        "cycling": count_interactions(person_indices, bicycle_indices),
        "riding_motorcycle": count_interactions(person_indices, motorcycle_indices),
        "in_car": count_interactions(person_indices, car_indices),
        "in_truck": count_interactions(person_indices, truck_indices),
        "in_bus": count_interactions(person_indices, bus_indices),
        "waiting_bus_stop": count_interactions(person_indices, sign_indices),
    }

    # Direct action counts
    direct_action_counts = {}
    for action_label in DIRECT_ACTION_LABELS:
        cnt = sum(1 for b in boxes_info if b["label"] == action_label)
        if cnt > 0:
            display = ACTION_DISPLAY.get(action_label, action_label)
            direct_action_counts[display] = cnt

    # ── Object entries with color ────────────────────────────────────────────
    object_entries = {}
    for i, b in enumerate(boxes_info):
        label = b["label"]
        if label in DIRECT_ACTION_LABELS:
            continue
        colors = detect_dominant_color(img, b["box"], n_colors=3)
        entry = {
            "id": i,
            "confidence": round(b["conf"], 3),
            "colors": colors,
        }
        object_entries.setdefault(label, []).append(entry)

    # ── Build objects section ────────────────────────────────────────────────
    objects_section = {}
    for label, instances in object_entries.items():
        objects_section[label] = {
            "count": len(instances),
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
    raw_door_boxes = [b for b in boxes_info if b["label"] == "door"]
    filtered_doors = suppress_contained_doors(
        raw_door_boxes, containment_threshold=0.80
    )

    door_info = []
    for i, b in enumerate(filtered_doors):
        direction = get_door_direction(b["box"], img_w)
        door_info.append(
            {
                "id": i,
                "confidence": round(b["conf"], 3),
                "direction": direction,
                "description": f"There is a door on your {direction}",
            }
        )

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
            "source": gps_source,
            "lat": lat,
            "lng": lng,
            "poi_result": poi_result,
        }

    floor_section = estimate_floor_from_altitude(altitude)

    # ── Assemble output ──────────────────────────────────────────────────────
    output = {
        "objects": objects_section,
        "actions": actions_section,
        "doors": door_info if door_info else None,
        "summary": summary_section,
        "gps": gps_section,
        "floor": floor_section,
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
            entry = objects.setdefault(
                label,
                {
                    "count": 0,
                    "detections_in_frames": 0,
                    "instances": [],
                },
            )
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

        for door in result.get("doors") or []:
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
    altitude: Optional[float] = None,
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
                        altitude=None,
                        gps_radius=gps_radius,
                    )
                    frame_results.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_sec": round(frame_index / fps, 3)
                            if fps > 0
                            else None,
                            "analysis": analysis,
                        }
                    )
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
            "floor": estimate_floor_from_altitude(altitude),
            "frames": frame_results,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Model cache ──────────────────────────────────────────────────────────────────

# Lazy-loads each model on first use; all three fit comfortably in memory (~5 MB each).
_model_cache: dict[str, "YOLO"] = {}


def _get_model(path: str) -> "YOLO":
    """Return the cached YOLO instance for *path*, loading it if necessary."""
    if path not in _model_cache:
        _model_cache[path] = YOLO(path)
    return _model_cache[path]


# ── GPS-based indoor/outdoor environment detection ───────────────────────────────

# Module-level cache for POI data (loaded once at startup)
_poi_cache: Optional[list[dict]] = None


def _load_pois() -> list[dict]:
    """Load building POIs from the CSV file (cached after first call)."""
    global _poi_cache
    if _poi_cache is not None:
        return _poi_cache
    pois: list[dict] = []
    try:
        with open(POI_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pois.append(
                    {
                        "name": row["name"],
                        "lat": float(row["lat"]),
                        "lng": float(row["lng"]),
                    }
                )
    except Exception:
        pass
    _poi_cache = pois
    return pois


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return great-circle distance in metres between two GPS points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def determine_environment(
    lat: float,
    lng: float,
    indoor_radius_m: float = DEFAULT_INDOOR_RADIUS_M,
) -> dict:
    """
    Decide whether the given GPS position is indoors or outdoors.

    Strategy:
        Find the nearest building POI from poi_buildings_english.csv.
        If the distance is ≤ indoor_radius_m → indoor, else → outdoor.

    Returns a dict:
        {
            "environment":   "indoor" | "outdoor",
            "nearest_poi":   str | None,
            "distance_m":    float | None,
            "threshold_m":   float,
        }
    """
    pois = _load_pois()
    if not pois:
        return {
            "environment": "outdoor",
            "nearest_poi": None,
            "distance_m": None,
            "threshold_m": indoor_radius_m,
        }

    nearest_poi = None
    nearest_dist = float("inf")
    for poi in pois:
        d = _haversine_m(lat, lng, poi["lat"], poi["lng"])
        if d < nearest_dist:
            nearest_dist = d
            nearest_poi = poi["name"]

    environment = "indoor" if nearest_dist <= indoor_radius_m else "outdoor"
    return {
        "environment": environment,
        "nearest_poi": nearest_poi,
        "distance_m": round(nearest_dist, 1),
        "threshold_m": indoor_radius_m,
    }


def resolve_model(
    mode: ModelMode,
    lat: Optional[float],
    lng: Optional[float],
    indoor_radius_m: float = DEFAULT_INDOOR_RADIUS_M,
) -> tuple["YOLO", dict]:
    """
    Resolve which YOLO model to use and return (model_instance, model_info_dict).

    model_info_dict keys:
        mode          – the ModelMode requested by the caller
        selected      – "indoor" | "outdoor" | "base"
        model_file    – filename of the loaded model
        reason        – human-readable explanation
    """
    if mode == ModelMode.INDOOR:
        path = MODEL_INDOOR_PATH
        selected = "indoor"
        reason = "Explicitly requested by caller (model=indoor)"

    elif mode == ModelMode.OUTDOOR:
        path = MODEL_OUTDOOR_PATH
        selected = "outdoor"
        reason = "Explicitly requested by caller (model=outdoor)"

    elif mode == ModelMode.BASE:
        path = MODEL_BASE_PATH
        selected = "base"
        reason = "Explicitly requested by caller (model=base)"

    else:  # ModelMode.AUTO
        if lat is not None and lng is not None:
            env_info = determine_environment(lat, lng, indoor_radius_m)
            env = env_info["environment"]
            if env == "indoor":
                path = MODEL_INDOOR_PATH
                selected = "indoor"
                reason = (
                    f"Auto: nearest POI '{env_info['nearest_poi']}' is "
                    f"{env_info['distance_m']} m away "
                    f"(≤ {indoor_radius_m} m threshold)"
                )
            else:
                path = MODEL_OUTDOOR_PATH
                selected = "outdoor"
                reason = (
                    f"Auto: nearest POI '{env_info['nearest_poi']}' is "
                    f"{env_info['distance_m']} m away "
                    f"(> {indoor_radius_m} m threshold)"
                )
        else:
            path = MODEL_OUTDOOR_PATH
            selected = "outdoor"
            reason = "Auto: no GPS provided, defaulting to outdoor model"

    model_info = {
        "mode": mode.value,
        "selected": selected,
        "model_file": os.path.basename(path),
        "reason": reason,
    }
    return _get_model(path), model_info


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


@app.on_event("startup")
def startup():
    # Eagerly pre-load the outdoor model and POI data at startup
    _get_model(MODEL_OUTDOOR_PATH)
    _load_pois()


# ── API endpoints ────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def api_analyze(
    image: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    altitude: Optional[float] = Form(None),
    gps_radius: Optional[float] = Form(500),
    sample_interval_sec: Optional[float] = Form(1.0),
    max_frames: Optional[int] = Form(30),
    model: Optional[str] = Form("auto"),
    indoor_radius_m: Optional[float] = Form(DEFAULT_INDOOR_RADIUS_M),
):
    """
    Analyze an uploaded image or video.

    - **image**: Image or video file. The field name stays `image` for
      backward compatibility.
    - **lat**: Optional latitude (decimal degrees). If not given, tries EXIF.
    - **lng**: Optional longitude (decimal degrees). If not given, tries EXIF.
    - **altitude**: Optional altitude above sea level in meters for floor estimate.
    - **gps_radius**: POI search radius in meters (default 500).
    - **sample_interval_sec**: Video frame sampling interval in seconds.
    - **max_frames**: Maximum sampled video frames to analyze.
    - **model**: Model selection — one of `auto` (default), `indoor`, `outdoor`, `base`.
      When `auto`, the model is chosen by GPS proximity to building POIs:
      if the nearest building is within `indoor_radius_m` metres → `indoor.pt`,
      otherwise → `outdoor.pt`.  `base` always uses `yolo26n.pt`.
    - **indoor_radius_m**: Distance threshold in metres to classify the user as
      indoors when `model=auto` (default {DEFAULT_INDOOR_RADIUS_M} m).
    """
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload file")

    # ── Resolve model ────────────────────────────────────────────────────────
    try:
        mode = ModelMode(model or "auto")
    except ValueError:
        valid = [m.value for m in ModelMode]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Must be one of: {valid}",
        )

    selected_model, model_info = resolve_model(
        mode=mode,
        lat=lat,
        lng=lng,
        indoor_radius_m=indoor_radius_m or DEFAULT_INDOOR_RADIUS_M,
    )

    # ── Detect media type ────────────────────────────────────────────────────
    content_type = (image.content_type or "").lower()
    filename = (image.filename or "").lower()
    video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
    video_suffix = os.path.splitext(filename)[1] if filename else ".mp4"
    is_video = content_type.startswith("video/") or filename.endswith(video_extensions)

    try:
        if is_video:
            result = analyze_video(
                video_bytes=contents,
                model=selected_model,
                lat=lat,
                lng=lng,
                altitude=altitude,
                gps_radius=gps_radius or 500,
                sample_interval_sec=sample_interval_sec or 1.0,
                max_frames=max_frames or 30,
                suffix=video_suffix if video_suffix in video_extensions else ".mp4",
            )
        else:
            result = analyze_image(
                image_bytes=contents,
                model=selected_model,
                lat=lat,
                lng=lng,
                altitude=altitude,
                gps_radius=gps_radius or 500,
            )
            result["media_type"] = "image"
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result["model_info"] = model_info
    return JSONResponse(content=result)


# ── React frontend ───────────────────────────────────────────────────────────────

app.mount(
    "/frontend",
    StaticFiles(directory=FRONTEND_DIR),
    name="frontend",
)


@app.get("/", response_class=FileResponse)
def web_demo():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── CLI entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("Starting Surrounding Awareness API on http://localhost:8000")
    print("  API:  POST /api/analyze")
    print("  Demo: GET  /")
    uvicorn.run(app, host="0.0.0.0", port=8000)
