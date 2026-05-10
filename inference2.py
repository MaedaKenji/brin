from ultralytics import YOLO
import os
import json
import cv2
import numpy as np
import math

from color import detect_dominant_color


output_dir = r"D:\Agus\BRIN\my_output"
os.makedirs(output_dir, exist_ok=True)


# ── Label groups ────────────────────────────────────────────────────────────────

# Person variants in the model
PERSON_LABELS = {'person', 'person sitting', 'person standing'}

# Action-like classes that are detected directly as a class (not inferred by proximity)
DIRECT_ACTION_LABELS = {'Walking', 'smiling', 'person eating', 'person sitting', 'person standing'}

# Non-action / physical objects (will appear in "objects" section)
OBJECT_LABELS = {'bicycle', 'bus', 'car', 'chair', 'clothe', 'motorcycle',
                 'shelter', 'sign', 'traffic light', 'truck', 'water bootle', 'person'}

# Interaction-based action labels (inferred by proximity)
INTERACTION_ACTIONS = {
    'cycling':            {'person': PERSON_LABELS, 'vehicle': {'bicycle'}},
    'riding_motorcycle':  {'person': PERSON_LABELS, 'vehicle': {'motorcycle'}},
    'in_car':             {'person': PERSON_LABELS, 'vehicle': {'car'}},
    'in_truck':           {'person': PERSON_LABELS, 'vehicle': {'truck'}},
    'in_bus':             {'person': PERSON_LABELS, 'vehicle': {'bus'}},
    'waiting_bus_stop':   {'person': PERSON_LABELS, 'vehicle': {'sign'}},
}

# Human-readable action labels for the JSON
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
    """Calculate Intersection over Union (IoU) between two boxes."""
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
    c1 = center(box1)
    c2 = center(box2)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def boxes_nearby(box1, box2, iou_threshold=0.05, dist_threshold=150):
    """
    Check if two boxes are nearby via IoU overlap or center-to-center distance.
    """
    iou = calculate_iou(box1, box2)
    if iou >= iou_threshold:
        return True
    return center_distance(box1, box2) < dist_threshold


def get_color(cls_id):
    """Consistent per-class box color for drawing."""
    np.random.seed(cls_id)
    return tuple(int(x) for x in np.random.randint(0, 255, 3))


# ── Model & inference ────────────────────────────────────────────────────────────

model = YOLO(r"D:\Agus\BRIN\runs\detect\train18\weights\best.pt")
folder_path = r"D:\Agus\BRIN\images_saya"

results = model.predict(
    source=folder_path,
    conf=0.1,
    iou=0.1,
    agnostic_nms=False
)


# ── Per-image processing ─────────────────────────────────────────────────────────

for result in results:
    filename = os.path.basename(result.path)
    img = result.orig_img.copy()

    boxes_info = []

    # ── Step 1: collect detections ───────────────────────────────────────────────
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        cls_id = int(cls)
        label  = result.names[cls_id]
        boxes_info.append({
            'box':    box.tolist(),
            'cls_id': cls_id,
            'label':  label,
            'conf':   float(conf),
        })

        x1, y1, x2, y2 = map(int, box)
        color = get_color(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ── Step 2: build index sets ─────────────────────────────────────────────────
    person_indices     = [i for i, b in enumerate(boxes_info) if b['label'] in PERSON_LABELS]
    bicycle_indices    = [i for i, b in enumerate(boxes_info) if b['label'] == 'bicycle']
    motorcycle_indices = [i for i, b in enumerate(boxes_info) if b['label'] == 'motorcycle']
    car_indices        = [i for i, b in enumerate(boxes_info) if b['label'] == 'car']
    truck_indices      = [i for i, b in enumerate(boxes_info) if b['label'] == 'truck']
    bus_indices        = [i for i, b in enumerate(boxes_info) if b['label'] == 'bus']
    sign_indices       = [i for i, b in enumerate(boxes_info) if b['label'] == 'sign']

    # ── Step 3: interaction (proximity) action counts ────────────────────────────
    def count_interactions(p_indices, v_indices):
        count = 0
        for p_idx in p_indices:
            for v_idx in v_indices:
                if boxes_nearby(boxes_info[p_idx]['box'], boxes_info[v_idx]['box']):
                    count += 1
                    break
        return count

    cycling_count            = count_interactions(person_indices, bicycle_indices)
    riding_motorcycle_count  = count_interactions(person_indices, motorcycle_indices)
    in_car_count             = count_interactions(person_indices, car_indices)
    in_truck_count           = count_interactions(person_indices, truck_indices)
    in_bus_count             = count_interactions(person_indices, bus_indices)
    waiting_bus_stop_count   = count_interactions(person_indices, sign_indices)

    # Direct action class counts
    direct_action_counts = {}
    for action_label in DIRECT_ACTION_LABELS:
        cnt = sum(1 for b in boxes_info if b['label'] == action_label)
        if cnt > 0:
            display = ACTION_DISPLAY.get(action_label, action_label)
            direct_action_counts[display] = cnt

    # ── Step 4: object entries with color properties ─────────────────────────────
    # Group by label, include per-instance color detection
    object_entries = {}   # label -> list of {id, confidence, colors}

    for i, b in enumerate(boxes_info):
        label = b['label']
        # Anything that is not a direct action label → treat as an object
        if label in DIRECT_ACTION_LABELS:
            continue

        colors = detect_dominant_color(img, b['box'], n_colors=3)
        entry = {
            "id":         i,
            "confidence": round(b['conf'], 3),
            "colors":     colors,
        }
        object_entries.setdefault(label, []).append(entry)

    # ── Step 5: build the structured JSON ────────────────────────────────────────

    # --- objects section ---
    objects_section = {}
    for label, instances in object_entries.items():
        # Aggregate dominant colors across all instances of this label
        all_colors = [c for inst in instances for c in inst["colors"]]
        from collections import Counter
        color_freq = Counter(all_colors)
        top_colors = [c for c, _ in color_freq.most_common(3)]

        objects_section[label] = {
            "count":            len(instances),
            # "color_properties": top_colors,
            "instances":        instances,
        }

    # --- actions section ---
    actions_section = {}

    # Interaction-based actions
    interaction_map = {
        'cycling':           cycling_count,
        'riding_motorcycle': riding_motorcycle_count,
        'in_car':            in_car_count,
        'in_truck':          in_truck_count,
        'in_bus':            in_bus_count,
        'waiting_bus_stop':  waiting_bus_stop_count,
    }
    for key, cnt in interaction_map.items():
        if cnt > 0:
            display = ACTION_DISPLAY[key]
            actions_section[display] = {"count": cnt}

    # Direct action classes
    for display, cnt in direct_action_counts.items():
        actions_section[display] = {"count": cnt}

    # --- summary section ---
    summary_section = {}
    for label, data in objects_section.items():
        summary_section[label] = data["count"]

    # Also add action totals in summary
    for action, data in actions_section.items():
        summary_section[action] = data["count"]

    # ── Step 6: assemble full JSON ───────────────────────────────────────────────
    output_json = {
        "file":    filename,
        "objects": objects_section,
        "actions": actions_section,
        "summary": summary_section,
    }

    # ── Console print ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"File: {filename}")
    print(f"Objects detected: {list(objects_section.keys())}")
    print(f"Actions detected: {list(actions_section.keys())}")
    print("Summary:")
    for k, v in summary_section.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    # ── Save outputs ─────────────────────────────────────────────────────────────
    out_img_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
    cv2.imwrite(out_img_path, img)

    json_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.json')
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
