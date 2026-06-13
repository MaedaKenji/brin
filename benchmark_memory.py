"""
benchmark_memory.py -- RAM & GPU VRAM benchmark for final3.py models.

Measures system RAM and GPU VRAM at each stage:
  1. Python baseline (imports only)
  2. After loading each YOLO model (indoor.pt / outdoor.pt / yolo26n.pt)
  3. During and after a single inference pass
  4. After a colour-detection pass (K-means, same as final3.py)

Run standalone (do NOT start uvicorn first):
    python benchmark_memory.py

Outputs a formatted report and a recommended minimum system spec.
"""

import gc
import os
import sys
import time
import textwrap

import cv2
import numpy as np
import psutil

# ── Optional GPU tracking ────────────────────────────────────────────────────
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None          # type: ignore[assignment]
    CUDA_AVAILABLE = False

# ── Paths (same as final3.py) ────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
MODEL_INDOOR_PATH  = os.path.join(BASE_DIR, "indoor.pt")
MODEL_OUTDOOR_PATH = os.path.join(BASE_DIR, "outdoor.pt")
MODEL_BASE_PATH    = os.path.join(BASE_DIR, "yolo26n.pt")

MODELS = [
    ("indoor",  MODEL_INDOOR_PATH),
    ("outdoor", MODEL_OUTDOOR_PATH),
    ("base",    MODEL_BASE_PATH),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

_PROC = psutil.Process(os.getpid())


def ram_mb() -> float:
    """Current process RSS (Resident Set Size) in MB."""
    return _PROC.memory_info().rss / 1024 ** 2


def system_ram() -> dict:
    """Snapshot of total system RAM state."""
    vm = psutil.virtual_memory()
    return {
        "total_mb":     vm.total     / 1024 ** 2,
        "available_mb": vm.available / 1024 ** 2,
        "used_mb":      vm.used      / 1024 ** 2,
        "percent":      vm.percent,
    }


def vram_mb() -> dict | None:
    """
    Current GPU VRAM usage in MB (allocated + reserved) for device 0.
    Returns None if CUDA is not available.
    """
    if not CUDA_AVAILABLE or torch is None:
        return None
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated(0) / 1024 ** 2,
        "reserved_mb":  torch.cuda.memory_reserved(0)  / 1024 ** 2,
        "total_mb":     torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
    }


def reset_vram_peak():
    if CUDA_AVAILABLE and torch is not None:
        torch.cuda.reset_peak_memory_stats(0)


def peak_vram_allocated_mb() -> float | None:
    if not CUDA_AVAILABLE or torch is None:
        return None
    return torch.cuda.max_memory_allocated(0) / 1024 ** 2


def make_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a synthetic BGR image with random content (no disk I/O needed)."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    # Draw a rough human silhouette so the model has something to detect
    cv2.rectangle(img, (280, 100), (360, 420), (200, 180, 160), -1)
    cv2.circle(img,   (320, 80),  40,          (220, 200, 180), -1)
    return img


def encode_image_to_bytes(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def section(title: str):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def fmt_vram(v: dict | None) -> str:
    if v is None:
        return "n/a (no CUDA GPU)"
    return (
        f"allocated {v['allocated_mb']:.1f} MB  /  "
        f"reserved {v['reserved_mb']:.1f} MB  /  "
        f"total {v['total_mb']:.0f} MB"
    )


# ── Main benchmark ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  benchmark_memory.py  --  final3.py RAM / VRAM Benchmark")
    print("=" * 60)

    # ── System info ──────────────────────────────────────────────────────────
    section("System Information")
    sram = system_ram()
    print(f"  OS            : {sys.platform}")
    print(f"  Python        : {sys.version.split()[0]}")
    print(f"  CPU count     : {psutil.cpu_count(logical=True)} logical cores")
    print(f"  Total RAM     : {sram['total_mb']:.0f} MB  ({sram['total_mb']/1024:.1f} GB)")
    print(f"  Available RAM : {sram['available_mb']:.0f} MB")

    if CUDA_AVAILABLE and torch is not None:
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU           : {props.name}")
        print(f"  GPU VRAM      : {props.total_memory / 1024**2:.0f} MB  ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("  GPU           : None / CUDA not available  (CPU inference only)")

    # ── Baseline ─────────────────────────────────────────────────────────────
    section("Baseline (after imports, before any model load)")
    gc.collect()
    baseline_ram = ram_mb()
    baseline_vram = vram_mb()
    print(f"  Process RAM   : {baseline_ram:.1f} MB")
    print(f"  VRAM          : {fmt_vram(baseline_vram)}")

    # ── Test image ───────────────────────────────────────────────────────────
    test_img    = make_test_image()
    test_bytes  = encode_image_to_bytes(test_img)
    img_h, img_w = test_img.shape[:2]
    print(f"\n  Test image    : {img_w}x{img_h} px  ({len(test_bytes)/1024:.1f} KB JPEG)")

    # ── Per-model benchmarks ─────────────────────────────────────────────────
    results: list[dict] = []

    for model_name, model_path in MODELS:
        if not os.path.exists(model_path):
            print(f"\n  [SKIP] {model_name}: file not found -> {model_path}")
            continue

        section(f"Model: {model_name}  ({os.path.basename(model_path)})")
        file_mb = os.path.getsize(model_path) / 1024 ** 2
        print(f"  Model file size : {file_mb:.1f} MB")

        # ── Load ─────────────────────────────────────────────────────────────
        gc.collect()
        reset_vram_peak()
        ram_before_load = ram_mb()

        from ultralytics import YOLO   # import here so baseline is clean
        t0 = time.perf_counter()
        model = YOLO(model_path)
        load_time = time.perf_counter() - t0

        ram_after_load  = ram_mb()
        vram_after_load = vram_mb()
        load_ram_delta  = ram_after_load - ram_before_load

        print(f"\n  [Load]")
        print(f"    Time          : {load_time:.2f} s")
        print(f"    RAM before    : {ram_before_load:.1f} MB")
        print(f"    RAM after     : {ram_after_load:.1f} MB  (+{load_ram_delta:.1f} MB)")
        print(f"    VRAM          : {fmt_vram(vram_after_load)}")

        # ── Warm-up (one silent pass to initialise CUDA kernels) ─────────────
        _ = model.predict(source=test_img, conf=0.1, iou=0.1,
                          agnostic_nms=False, verbose=False)
        gc.collect()
        reset_vram_peak()

        # ── Inference (N runs, take min/avg/max) ──────────────────────────────
        N = 5
        times       = []
        ram_peaks   = []
        vram_peaks  = []

        print(f"\n  [Inference  x{N} runs on {img_w}x{img_h} image]")

        for i in range(N):
            reset_vram_peak()
            ram_pre = ram_mb()

            t0 = time.perf_counter()
            results_yolo = model.predict(source=test_img, conf=0.1, iou=0.1,
                                         agnostic_nms=False, verbose=False)
            elapsed = time.perf_counter() - t0

            ram_post = ram_mb()
            peak_v   = peak_vram_allocated_mb()

            times.append(elapsed)
            ram_peaks.append(ram_post - ram_pre)
            if peak_v is not None:
                vram_peaks.append(peak_v)

            detections = len(results_yolo[0].boxes)
            print(f"    Run {i+1}: {elapsed*1000:.1f} ms  |  "
                  f"RAM D {ram_post - ram_pre:+.1f} MB  |  "
                  f"detections: {detections}")

        print(f"\n    -- Summary --")
        print(f"    Avg latency   : {sum(times)/N*1000:.1f} ms")
        print(f"    Min latency   : {min(times)*1000:.1f} ms")
        print(f"    Max latency   : {max(times)*1000:.1f} ms")
        print(f"    Avg RAM D     : {sum(ram_peaks)/N:+.1f} MB per inference")
        if vram_peaks:
            print(f"    Peak VRAM     : {max(vram_peaks):.1f} MB  (across all runs)")

        # ── Colour-detection pass (mirrors final3.py post-processing) ────────
        from color import detect_dominant_color   # same import as final3.py
        gc.collect()
        ram_pre_color = ram_mb()
        dummy_boxes = [[50, 50, 200, 300], [250, 80, 400, 350]]
        for box in dummy_boxes:
            detect_dominant_color(test_img, box, n_colors=3)
        ram_color_delta = ram_mb() - ram_pre_color
        print(f"\n  [Color detection (K-means, 2 boxes)]")
        print(f"    RAM D         : {ram_color_delta:+.1f} MB")

        # ── Unload model ─────────────────────────────────────────────────────
        del model
        gc.collect()
        if CUDA_AVAILABLE and torch is not None:
            torch.cuda.empty_cache()
        ram_after_unload = ram_mb()
        print(f"\n  [After model unload]  RAM: {ram_after_unload:.1f} MB")

        results.append({
            "name":           model_name,
            "file_mb":        file_mb,
            "load_ram_mb":    load_ram_delta,
            "avg_latency_ms": sum(times) / N * 1000,
            "min_latency_ms": min(times) * 1000,
            "max_latency_ms": max(times) * 1000,
            "peak_vram_mb":   max(vram_peaks) if vram_peaks else None,
            "ram_after_load": ram_after_load,
        })

    # ── System requirements summary ──────────────────────────────────────────
    section("System Requirements Summary")

    if not results:
        print("  No models were benchmarked (check file paths).")
        return

    # final3.py can cache all 3 models in memory simultaneously (lazy-loaded but
    # retained).  Worst-case RAM = baseline + sum of all model deltas + headroom.
    worst_ram = baseline_ram + sum(r["load_ram_mb"] for r in results) + 256  # 256 MB OS/app headroom
    worst_vram = max(
        (r["peak_vram_mb"] for r in results if r["peak_vram_mb"] is not None),
        default=None,
    )

    print(f"\n  {'Model':<10} {'File':>8}  {'Load dRAM':>10}  "
          f"{'Avg ms':>8}  {'Min ms':>8}  {'Max ms':>8}  {'Peak VRAM':>10}")
    print(f"  {'-'*10} {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in results:
        vram_str = f"{r['peak_vram_mb']:.1f} MB" if r["peak_vram_mb"] else "CPU only"
        print(f"  {r['name']:<10} {r['file_mb']:>6.1f}MB  "
              f"{r['load_ram_mb']:>+9.1f}MB  "
              f"{r['avg_latency_ms']:>7.1f}ms  "
              f"{r['min_latency_ms']:>7.1f}ms  "
              f"{r['max_latency_ms']:>7.1f}ms  "
              f"{vram_str:>10}")

    print()
    _SEP = "+" + "-" * 58 + "+"
    def _row(txt=""):
        print(("| " + txt).ljust(59) + "|")
    print(_SEP)
    _row("RECOMMENDED MINIMUM SYSTEM SPECS")
    print(_SEP)
    _row("RAM (all 3 models cached simultaneously)")
    _row(f"  Minimum  : {worst_ram:>6.0f} MB  ({worst_ram/1024:.1f} GB)")
    _row(f"  Baseline : {baseline_ram:>6.1f} MB  (Python + imports)")
    _row()
    _row("GPU VRAM (if using CUDA)")
    if worst_vram is not None:
        vram_min = worst_vram + 256
        _row(f"  Minimum   : {vram_min:>6.0f} MB  ({vram_min/1024:.1f} GB)")
        _row(f"  Peak infer: {worst_vram:>5.1f} MB")
    else:
        _row("  GPU : not used (CPU-only inference detected)")
        _row("  To enable GPU, install PyTorch with CUDA support.")
    avg_ms = sum(r["avg_latency_ms"] for r in results) / len(results)
    fps    = 1000 / avg_ms if avg_ms > 0 else 0
    _row()
    _row("Inference speed (single 640x480 image)")
    _row(f"  Avg latency : {avg_ms:>6.1f} ms  (~{fps:.1f} fps theoretical)")
    _row()
    _row("Notes")
    _row("  final3.py caches all 3 models after first use.")
    _row("  Uvicorn adds ~30-50 MB overhead per worker.")
    _row("  Video analysis buffers frames; budget +200 MB.")
    print(_SEP)


if __name__ == "__main__":
    main()
