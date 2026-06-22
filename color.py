import cv2
import numpy as np


# Named color palette (BGR format for OpenCV)
COLOR_PALETTE = {
    "red":    (0,   0,   200),
    "orange": (0,   128, 255),
    "yellow": (0,   220, 220),
    "green":  (0,   180, 0),
    "cyan":   (200, 200, 0),
    "blue":   (200, 0,   0),
    "purple": (180, 0,   180),
    "pink":   (180, 0,   120),
    "white":  (240, 240, 240),
    "gray":   (128, 128, 128),
    "black":  (20,  20,  20),
    "brown":  (30,  80,  140),
    "silver": (192, 192, 192),
}


def _closest_color_name(bgr):
    """Return the name of the closest color in COLOR_PALETTE to the given BGR tuple."""
    min_dist = float("inf")
    best_name = "unknown"
    for name, ref_bgr in COLOR_PALETTE.items():
        dist = np.sqrt(sum((int(a) - int(b)) ** 2 for a, b in zip(bgr, ref_bgr)))
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name


def detect_dominant_color(img, box, n_colors=3, max_dim=64):
    """
    Detect the dominant color(s) inside a bounding box using K-means clustering.

    Parameters
    ----------
    img     : BGR image (numpy array)
    box     : [x1, y1, x2, y2]
    n_colors: number of dominant colors to extract
    max_dim : resize crop to at most this dimension for speed

    Returns
    -------
    list of color name strings, ordered by dominance (most dominant first)
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return ["unknown"]

    # Resize for speed
    ch, cw = crop.shape[:2]
    if ch > max_dim or cw > max_dim:
        scale = max_dim / max(ch, cw)
        crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))

    # Reshape to list of pixels
    pixels = crop.reshape(-1, 3).astype(np.float32)

    if len(pixels) < n_colors:
        n_colors = max(1, len(pixels))

    # K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_colors, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
    )

    # Sort clusters by frequency (most dominant first)
    counts = np.bincount(labels.flatten())
    sorted_indices = np.argsort(-counts)

    color_names = []
    seen = set()
    for idx in sorted_indices:
        bgr = tuple(int(v) for v in centers[idx])
        name = _closest_color_name(bgr)
        if name not in seen:
            seen.add(name)
            color_names.append(name)

    return color_names if color_names else ["unknown"]
