from ultralytics import YOLO
import os
import json
import cv2
import numpy as np
import math


output_dir = r"D:\Agus\BRIN\my_output"
os.makedirs(output_dir, exist_ok=True)

# 0: car
#   1: bus
#   2: person
#   3: handbag
#   4: potted plant
#   5: book
#   6: bottle
#   7: dining table
#   8: vase
#   9: truck
#   10: traffic light
#   11: fire hydrant
#   12: umbrella
#   13: bicycle
#   14: horse
#   15: boat
#   16: kite
#   17: train
#   18: frisbee
#   19: sports ball
#   20: backpack
#   21: teddy bear
#   22: tennis racket
#   23: skateboard
#   24: airplane
#   25: motorcycle
#   26: pizza
#   27: donut
#   28: cake
#   29: clock
#   30: chair
#   31: bench
#   32: surfboard
#   33: tie
#   34: stop sign
#   35: cell phone
#   36: parking meter
#   37: dog
#   38: cow
#   39: suitcase
#   40: carrot
#   41: elephant
#   42: banana
#   43: cat
#   44: refrigerator
#   45: sheep
#   46: bird
#   47: cup
#   48: microwave
#   49: baseball glove
#   50: knife
#   51: sandwich

# names: ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22', '23', '24', '25', '26', '27', '28', '29', '3', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '6', '7', '8', '9', 'chair', 'clothe', 'doll', 'person sitting', 'person standing', 'smiling', 'water bootle']

class_map = {
    0: 'car',              # '0'
    1: 'bus',              # '1'
    2: 'traffic light',    # '10'
    3: 'fire hydrant',     # '11'
    4: 'umbrella',         # '12'
    5: 'bicycle',          # '13'
    6: 'horse',            # '14'
    7: 'boat',             # '15'
    8: 'kite',             # '16'
    9: 'train',            # '17'
    10: 'frisbee',         # '18'
    11: 'sports ball',     # '19'
    12: 'backpack',        # '20'
    13: 'tennis racket',   # '22'
    14: 'skateboard',      # '23'
    15: 'airplane',        # '24'
    16: 'motorcycle',      # '25'
    17: 'pizza',           # '26'
    18: 'donut',           # '27'
    19: 'cake',            # '28'
    20: 'clock',           # '29'
    21: 'handbag',         # '3'
    22: 'bench',           # '31'
    23: 'surfboard',       # '32'
    24: 'tie',             # '33'
    25: 'stop sign',       # '34'
    26: 'cell phone',      # '35'
    27: 'parking meter',   # '36'
    28: 'dog',             # '37'
    29: 'cow',             # '38'
    30: 'suitcase',        # '39'
    31: 'potted plant',    # '4'
    32: 'carrot',          # '40'
    33: 'elephant',        # '41'
    34: 'banana',          # '42'
    35: 'cat',             # '43'
    36: 'refrigerator',    # '44'
    37: 'sheep',           # '45'
    38: 'bird',            # '46'
    39: 'cup',             # '47'
    40: 'microwave',       # '48'
    41: 'baseball glove',  # '49'
    42: 'book',            # '5'
    43: 'knife',           # '50'
    44: 'sandwich',        # '51'
    45: 'bottle',          # '6'
    46: 'dining table',    # '7'
    47: 'vase',            # '8'
    48: 'truck',           # '9'
    49: 'chair',           # 'chair'
    50: 'clothe',          # 'clothe'
    51: 'doll',            # 'doll'
    52: 'person sitting',  # 'person sitting'
    53: 'person standing', # 'person standing'
    54: 'smiling',         # 'smiling'
    55: 'water bootle'     # 'water bootle'
}

# Labels that represent a person (to match all person variants in class_map)
PERSON_LABELS = {'person', 'person sitting', 'person standing'}


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        inter_area = 0
    else:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union area
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
    Check if two boxes are nearby.
    First checks IoU overlap; if not overlapping enough, falls back to
    center-to-center pixel distance.
    - iou_threshold: minimum IoU to consider boxes overlapping/nearby.
    - dist_threshold: max pixel distance between centers as a fallback.
    """
    iou = calculate_iou(box1, box2)
    if iou >= iou_threshold:
        return True

    # Fallback: center pixel distance
    return center_distance(box1, box2) < dist_threshold


model = YOLO(r"D:\Agus\BRIN\runs\detect\train10\weights\best.pt")
folder_path = r"D:\Agus\BRIN\images_saya"

results = model.predict(
    source=folder_path,
    conf=0.3,
    iou=0.1,
    agnostic_nms=False
)

for result in results:
    filename = os.path.basename(result.path)
    img = result.orig_img.copy()

    detected_classes = []
    person_count = 0

    # Store box info for interaction detection
    boxes_info = []

    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        cls_id = int(cls)
        label = class_map.get(cls_id, str(cls_id))
        detected_classes.append(label)
        boxes_info.append({'box': box.tolist(), 'cls_id': cls_id, 'label': label, 'conf': float(conf)})

        if label in PERSON_LABELS:
            person_count += 1

        # Bounding box
        x1, y1, x2, y2 = map(int, box)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        text = f"{label} {conf:.2f}"
        cv2.putText(
            img, text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    print(f"\n{'='*60}")
    print(f"File: {filename}")
    print(f"Detected: {', '.join(detected_classes) if detected_classes else 'None'}")
    print(f"Total persons: {person_count}")

    # === INTERACTION DETECTION ===

    # Gather indices by label group
    person_indices    = [i for i, info in enumerate(boxes_info) if info['label'] in PERSON_LABELS]
    bicycle_indices   = [i for i, info in enumerate(boxes_info) if info['label'] == 'bicycle']
    motorcycle_indices= [i for i, info in enumerate(boxes_info) if info['label'] == 'motorcycle']
    car_indices       = [i for i, info in enumerate(boxes_info) if info['label'] == 'car']
    truck_indices     = [i for i, info in enumerate(boxes_info) if info['label'] == 'truck']
    bus_indices       = [i for i, info in enumerate(boxes_info) if info['label'] == 'bus']

    # 1. CYCLING: Person near a Bicycle
    cycling_count = 0
    for p_idx in person_indices:
        for b_idx in bicycle_indices:
            if boxes_nearby(boxes_info[p_idx]['box'], boxes_info[b_idx]['box']):
                cycling_count += 1
                break  # Count this person as cycling only once

    print(f"Persons cycling: {cycling_count}")

    # 2. RIDING MOTORCYCLE: Person near a Motorcycle
    riding_motorcycle_count = 0
    for p_idx in person_indices:
        for m_idx in motorcycle_indices:
            if boxes_nearby(boxes_info[p_idx]['box'], boxes_info[m_idx]['box']):
                riding_motorcycle_count += 1
                break

    print(f"Persons riding motorcycle: {riding_motorcycle_count}")

    # 3. IN CAR: Person near a Car
    in_car_count = 0
    for p_idx in person_indices:
        for c_idx in car_indices:
            if boxes_nearby(boxes_info[p_idx]['box'], boxes_info[c_idx]['box']):
                in_car_count += 1
                break

    print(f"Persons in/near car: {in_car_count}")

    # 4. IN TRUCK: Person near a Truck
    in_truck_count = 0
    for p_idx in person_indices:
        for t_idx in truck_indices:
            if boxes_nearby(boxes_info[p_idx]['box'], boxes_info[t_idx]['box']):
                in_truck_count += 1
                break

    print(f"Persons in/near truck: {in_truck_count}")

    # 5. IN BUS: Person near a Bus
    in_bus_count = 0
    for p_idx in person_indices:
        for b_idx in bus_indices:
            if boxes_nearby(boxes_info[p_idx]['box'], boxes_info[b_idx]['box']):
                in_bus_count += 1
                break

    print(f"Persons in/near bus: {in_bus_count}")

    print(f"{'='*60}")

    # Save annotated image
    out_img_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
    cv2.imwrite(out_img_path, img)

    bicycle_count = len(bicycle_indices)

    # Save JSON summary
    summary = {
        'file': filename,
        'detected': detected_classes,
        'person': int(person_count),
        'bicycle': int(bicycle_count),
        'cycling': int(cycling_count),
        'riding_motorcycle': int(riding_motorcycle_count),
        'in_car': int(in_car_count),
        'in_truck': int(in_truck_count),
        'in_bus': int(in_bus_count),
    }

    json_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
