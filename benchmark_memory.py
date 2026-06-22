"""
benchmark_memory.py -- Comprehensive RAM, CPU & GPU benchmark for final4.py models.

Measures:
  Part 1: Per-model resource usage
    - RAM (RSS) before/after model load and during inference
    - CPU utilisation % during inference
    - GPU VRAM allocated/reserved/peak during inference
    - GPU utilisation % (via pynvml if available)
    - Inference latency (min/avg/max over N runs)

  Part 2: Auto-switching overhead
    - Cost of switching between indoor.pt ↔ outdoor.pt (simulating final4.py auto mode)
    - Cache-clear impact (gc.collect + torch.cuda.empty_cache)
    - I/O reload penalty when model is evicted from cache

  Part 3: CPU-only vs GPU comparison
    - Forces CPU-only inference (device='cpu') for each model
    - Compares latency and RAM vs GPU results from Part 1

Run standalone (do NOT start uvicorn first):
    python benchmark_memory.py

Outputs a formatted report to console and saves results to benchmark_results.json.
"""

import gc
import json
import os
import sys
import time
import textwrap
from datetime import datetime

import cv2
import numpy as np
import psutil

# ── Optional GPU tracking ────────────────────────────────────────────────────
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    CUDA_AVAILABLE = False

# ── Optional NVML for GPU utilisation % ──────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# ── Paths (same as final4.py) ────────────────────────────────────────────────
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


def cpu_percent_snapshot(interval: float = 0.1) -> float:
    """Get current process CPU utilisation % (blocking for `interval` seconds)."""
    return _PROC.cpu_percent(interval=interval)


def system_cpu_percent(interval: float = 0.1) -> float:
    """Get system-wide CPU utilisation %."""
    return psutil.cpu_percent(interval=interval)


def vram_mb() -> dict | None:
    """Current GPU VRAM usage in MB for device 0."""
    if not CUDA_AVAILABLE or torch is None:
        return None
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated(0) / 1024 ** 2,
        "reserved_mb":  torch.cuda.memory_reserved(0)  / 1024 ** 2,
        "total_mb":     torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
    }


def gpu_utilisation_pct() -> float | None:
    """Get GPU utilisation % via NVML. Returns None if unavailable."""
    if not NVML_AVAILABLE:
        return None
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
        return util.gpu
    except Exception:
        return None


def gpu_memory_utilisation_pct() -> float | None:
    """Get GPU memory utilisation % via NVML."""
    if not NVML_AVAILABLE:
        return None
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
        return util.memory
    except Exception:
        return None


def reset_vram_peak():
    if CUDA_AVAILABLE and torch is not None:
        torch.cuda.reset_peak_memory_stats(0)


def peak_vram_allocated_mb() -> float | None:
    if not CUDA_AVAILABLE or torch is None:
        return None
    return torch.cuda.max_memory_allocated(0) / 1024 ** 2


def make_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a synthetic BGR image with rough shapes for detection."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    # Draw a rough human silhouette so the model has something to detect
    cv2.rectangle(img, (280, 100), (360, 420), (200, 180, 160), -1)
    cv2.circle(img,   (320, 80),  40,          (220, 200, 180), -1)
    # Add more objects to make detection more realistic
    cv2.rectangle(img, (50, 200), (150, 400), (100, 120, 140), -1)   # another person
    cv2.rectangle(img, (400, 300), (600, 460), (150, 150, 200), -1)  # vehicle-like
    return img


def encode_image_to_bytes(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def subsection(title: str):
    print(f"\n  {'-' * 60}")
    print(f"    {title}")
    print(f"  {'-' * 60}")


def fmt_vram(v: dict | None) -> str:
    if v is None:
        return "n/a (no CUDA GPU)"
    return (
        f"allocated {v['allocated_mb']:.1f} MB  /  "
        f"reserved {v['reserved_mb']:.1f} MB  /  "
        f"total {v['total_mb']:.0f} MB"
    )


# ──────────────────────────────────────────────────────────────────────────────
#  PART 1: Per-Model Resource Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_single_model(
    model_name: str,
    model_path: str,
    test_img: np.ndarray,
    device: str = "",   # "" = auto (GPU if available), "cpu" = force CPU
    n_runs: int = 5,
) -> dict | None:
    """
    Benchmark a single YOLO model: load time, RAM, VRAM, CPU%, GPU%, latency.

    Args:
        device: "" for auto (GPU if available), "cpu" for forced CPU mode.
    """
    if not os.path.exists(model_path):
        print(f"\n  [SKIP] {model_name}: file not found -> {model_path}")
        return None

    device_label = device if device else ("GPU" if CUDA_AVAILABLE else "CPU")
    subsection(f"Model: {model_name}  ({os.path.basename(model_path)})  [{device_label}]")

    file_mb = os.path.getsize(model_path) / 1024 ** 2
    print(f"    Model file size : {file_mb:.1f} MB")

    img_h, img_w = test_img.shape[:2]

    # ── Load ─────────────────────────────────────────────────────────────
    gc.collect()
    if CUDA_AVAILABLE and torch is not None:
        torch.cuda.empty_cache()
    reset_vram_peak()
    ram_before_load = ram_mb()

    from ultralytics import YOLO
    t0 = time.perf_counter()
    model = YOLO(model_path)
    load_time = time.perf_counter() - t0

    ram_after_load  = ram_mb()
    vram_after_load = vram_mb()
    load_ram_delta  = ram_after_load - ram_before_load

    print(f"\n    [Load]")
    print(f"      Time          : {load_time:.2f} s")
    print(f"      RAM before    : {ram_before_load:.1f} MB")
    print(f"      RAM after     : {ram_after_load:.1f} MB  (+{load_ram_delta:.1f} MB)")
    print(f"      VRAM          : {fmt_vram(vram_after_load)}")

    # ── Warm-up (initialise CUDA kernels / graph) ────────────────────────
    predict_kwargs = dict(
        source=test_img, conf=0.1, iou=0.1,
        agnostic_nms=False, verbose=False,
    )
    if device:
        predict_kwargs["device"] = device

    _ = model.predict(**predict_kwargs)
    gc.collect()
    reset_vram_peak()

    # Prime the CPU measurement so the first reading isn't zero
    cpu_percent_snapshot(interval=0.05)

    # ── Inference (N runs) ───────────────────────────────────────────────
    times       = []
    ram_peaks   = []
    vram_peaks  = []
    cpu_usages  = []
    gpu_usages  = []
    gpu_mem_usages = []

    print(f"\n    [Inference x{n_runs} runs on {img_w}x{img_h} image]")

    for i in range(n_runs):
        reset_vram_peak()
        ram_pre = ram_mb()

        # Start CPU measurement window
        _PROC.cpu_percent()  # reset counter
        gpu_before = gpu_utilisation_pct()

        t0 = time.perf_counter()
        results_yolo = model.predict(**predict_kwargs)
        elapsed = time.perf_counter() - t0

        cpu_pct = _PROC.cpu_percent()
        gpu_after = gpu_utilisation_pct()
        gpu_mem = gpu_memory_utilisation_pct()

        ram_post = ram_mb()
        peak_v   = peak_vram_allocated_mb()

        times.append(elapsed)
        ram_peaks.append(ram_post - ram_pre)
        cpu_usages.append(cpu_pct)

        if peak_v is not None:
            vram_peaks.append(peak_v)
        if gpu_after is not None:
            gpu_usages.append(gpu_after)
        if gpu_mem is not None:
            gpu_mem_usages.append(gpu_mem)

        detections = len(results_yolo[0].boxes)
        gpu_str = f"GPU {gpu_after}%" if gpu_after is not None else "GPU n/a"
        print(f"      Run {i+1}: {elapsed*1000:.1f} ms  |  "
              f"CPU {cpu_pct:.0f}%  |  {gpu_str}  |  "
              f"RAM Δ {ram_post - ram_pre:+.1f} MB  |  "
              f"detections: {detections}")

    # ── Summary ──────────────────────────────────────────────────────────
    avg_latency = sum(times) / n_runs * 1000
    min_latency = min(times) * 1000
    max_latency = max(times) * 1000
    avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    avg_gpu = sum(gpu_usages) / len(gpu_usages) if gpu_usages else None
    avg_gpu_mem = sum(gpu_mem_usages) / len(gpu_mem_usages) if gpu_mem_usages else None

    print(f"\n    -- Summary --")
    print(f"    Avg latency   : {avg_latency:.1f} ms")
    print(f"    Min latency   : {min_latency:.1f} ms")
    print(f"    Max latency   : {max_latency:.1f} ms")
    print(f"    Avg RAM Δ     : {sum(ram_peaks)/n_runs:+.1f} MB per inference")
    print(f"    Avg CPU usage : {avg_cpu:.1f}%")
    if vram_peaks:
        print(f"    Peak VRAM     : {max(vram_peaks):.1f} MB")
    if avg_gpu is not None:
        print(f"    Avg GPU util  : {avg_gpu:.1f}%")
    if avg_gpu_mem is not None:
        print(f"    Avg GPU mem % : {avg_gpu_mem:.1f}%")

    # ── Colour-detection pass (mirrors final4.py post-processing) ────────
    from color import detect_dominant_color
    gc.collect()
    ram_pre_color = ram_mb()
    dummy_boxes = [[50, 50, 200, 300], [250, 80, 400, 350]]
    for box in dummy_boxes:
        detect_dominant_color(test_img, box, n_colors=3)
    ram_color_delta = ram_mb() - ram_pre_color
    print(f"\n    [Color detection (K-means, 2 boxes)]")
    print(f"      RAM Δ       : {ram_color_delta:+.1f} MB")

    # ── Unload model ─────────────────────────────────────────────────────
    del model
    gc.collect()
    if CUDA_AVAILABLE and torch is not None:
        torch.cuda.empty_cache()
    ram_after_unload = ram_mb()
    print(f"\n    [After model unload]  RAM: {ram_after_unload:.1f} MB")

    return {
        "name":              model_name,
        "device":            device_label,
        "file_mb":           file_mb,
        "load_time_s":       load_time,
        "load_ram_delta_mb": load_ram_delta,
        "avg_latency_ms":    avg_latency,
        "min_latency_ms":    min_latency,
        "max_latency_ms":    max_latency,
        "avg_cpu_pct":       avg_cpu,
        "avg_gpu_pct":       avg_gpu,
        "avg_gpu_mem_pct":   avg_gpu_mem,
        "peak_vram_mb":      max(vram_peaks) if vram_peaks else None,
        "ram_after_load_mb": ram_after_load,
        "color_ram_delta_mb": ram_color_delta,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  PART 2: Auto-Switching Overhead
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_auto_switching(test_img: np.ndarray, n_switches: int = 10) -> dict:
    """
    Simulate final4.py auto-switching between indoor.pt and outdoor.pt.

    Measures:
    - Cached switch: switching models that are already in memory (no reload)
    - Cold switch: after clearing the model from cache (forces reload + I/O)
    - gc.collect + torch.cuda.empty_cache overhead
    """
    section("Part 2: Auto-Switching Overhead (indoor ↔ outdoor)")

    from ultralytics import YOLO

    predict_kwargs = dict(source=test_img, conf=0.1, iou=0.1,
                          agnostic_nms=False, verbose=False)

    # Pre-load both models to simulate final4.py's _model_cache
    print("\n  Pre-loading both models into cache...")
    model_indoor = YOLO(MODEL_INDOOR_PATH)
    model_outdoor = YOLO(MODEL_OUTDOOR_PATH)

    # Warm up both
    _ = model_indoor.predict(**predict_kwargs)
    _ = model_outdoor.predict(**predict_kwargs)
    gc.collect()

    ram_both_cached = ram_mb()
    vram_both_cached = vram_mb()
    print(f"  RAM with both models cached: {ram_both_cached:.1f} MB")
    print(f"  VRAM with both cached      : {fmt_vram(vram_both_cached)}")

    # ── Scenario A: Cached switching (no reload) ─────────────────────────
    subsection("Scenario A: Cached switching (models stay in memory)")
    print(f"    Alternating indoor ↔ outdoor for {n_switches} switches...\n")

    cached_switch_times = []
    cached_switch_ram_deltas = []
    models_cycle = [model_indoor, model_outdoor]

    # Prime CPU counter
    _PROC.cpu_percent()

    for i in range(n_switches):
        current_model = models_cycle[i % 2]
        model_label = "indoor" if i % 2 == 0 else "outdoor"

        ram_pre = ram_mb()
        reset_vram_peak()

        t0 = time.perf_counter()
        _ = current_model.predict(**predict_kwargs)
        elapsed = time.perf_counter() - t0

        cpu_pct = _PROC.cpu_percent()
        ram_post = ram_mb()

        cached_switch_times.append(elapsed)
        cached_switch_ram_deltas.append(ram_post - ram_pre)

        print(f"    Switch {i+1:2d} → {model_label:7s}: "
              f"{elapsed*1000:.1f} ms  |  CPU {cpu_pct:.0f}%  |  "
              f"RAM Δ {ram_post - ram_pre:+.1f} MB")

    avg_cached = sum(cached_switch_times) / len(cached_switch_times) * 1000
    print(f"\n    Avg cached switch latency: {avg_cached:.1f} ms")
    print(f"    Avg RAM Δ per switch     : {sum(cached_switch_ram_deltas)/len(cached_switch_ram_deltas):+.1f} MB")

    # ── Scenario B: Cold switching (evict + reload) ──────────────────────
    subsection("Scenario B: Cold switching (evict model, force reload from disk)")

    cold_switch_times = []
    cold_reload_times = []
    cold_gc_times     = []

    for i in range(min(n_switches, 6)):  # fewer iterations since reload is expensive
        model_label = "indoor" if i % 2 == 0 else "outdoor"
        model_path = MODEL_INDOOR_PATH if i % 2 == 0 else MODEL_OUTDOOR_PATH

        # Simulate evicting the model (as if final4.py cleared _model_cache)
        ram_pre_gc = ram_mb()
        t_gc0 = time.perf_counter()

        # Delete and collect — measure gc + cache clear overhead
        if i % 2 == 0:
            del model_indoor
        else:
            del model_outdoor
        gc.collect()
        if CUDA_AVAILABLE and torch is not None:
            torch.cuda.empty_cache()

        t_gc = time.perf_counter() - t_gc0
        ram_post_gc = ram_mb()
        cold_gc_times.append(t_gc)

        # Reload from disk
        t_reload0 = time.perf_counter()
        reloaded = YOLO(model_path)
        t_reload = time.perf_counter() - t_reload0
        cold_reload_times.append(t_reload)

        # Inference after reload
        t_inf0 = time.perf_counter()
        _ = reloaded.predict(**predict_kwargs)
        t_inf = time.perf_counter() - t_inf0

        total = t_gc + t_reload + t_inf
        cold_switch_times.append(total)

        print(f"    Cold switch {i+1} → {model_label:7s}: "
              f"gc {t_gc*1000:.1f}ms + "
              f"reload {t_reload*1000:.1f}ms + "
              f"infer {t_inf*1000:.1f}ms = "
              f"total {total*1000:.1f}ms  |  "
              f"RAM freed by gc: {ram_pre_gc - ram_post_gc:+.1f} MB")

        # Restore for next iteration
        if i % 2 == 0:
            model_indoor = reloaded
        else:
            model_outdoor = reloaded

    avg_cold = sum(cold_switch_times) / len(cold_switch_times) * 1000
    avg_gc = sum(cold_gc_times) / len(cold_gc_times) * 1000
    avg_reload = sum(cold_reload_times) / len(cold_reload_times) * 1000

    print(f"\n    Avg cold switch total    : {avg_cold:.1f} ms")
    print(f"    Avg gc+cache clear       : {avg_gc:.1f} ms")
    print(f"    Avg model reload (I/O)   : {avg_reload:.1f} ms")
    print(f"    Overhead vs cached switch: +{avg_cold - avg_cached:.1f} ms ({avg_cold/avg_cached:.1f}x slower)")

    # Cleanup
    del model_indoor, model_outdoor
    gc.collect()
    if CUDA_AVAILABLE and torch is not None:
        torch.cuda.empty_cache()

    return {
        "cached_switch_avg_ms":  avg_cached,
        "cached_switch_times":   [t * 1000 for t in cached_switch_times],
        "cold_switch_avg_ms":    avg_cold,
        "cold_gc_avg_ms":        avg_gc,
        "cold_reload_avg_ms":    avg_reload,
        "cold_switch_times":     [t * 1000 for t in cold_switch_times],
        "ram_both_cached_mb":    ram_both_cached,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  benchmark_memory.py  --  final4.py Comprehensive Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── System info ──────────────────────────────────────────────────────
    section("System Information")
    sram = system_ram()
    print(f"  OS            : {sys.platform}")
    print(f"  Python        : {sys.version.split()[0]}")
    cpu_count = psutil.cpu_count(logical=True)
    print(f"  CPU count     : {cpu_count} logical cores")
    print(f"  Total RAM     : {sram['total_mb']:.0f} MB  ({sram['total_mb']/1024:.1f} GB)")
    print(f"  Available RAM : {sram['available_mb']:.0f} MB")
    print(f"  System load   : {sram['percent']:.1f}%")

    gpu_name = None
    gpu_vram_total = None
    if CUDA_AVAILABLE and torch is not None:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_vram_total = props.total_memory / 1024**2
        print(f"  GPU           : {gpu_name}")
        print(f"  GPU VRAM      : {gpu_vram_total:.0f} MB  ({gpu_vram_total / 1024:.1f} GB)")
        print(f"  CUDA version  : {torch.version.cuda}")
        print(f"  PyTorch       : {torch.__version__}")
    else:
        print("  GPU           : None / CUDA not available  (CPU inference only)")

    if NVML_AVAILABLE:
        print(f"  NVML          : available (real-time GPU utilisation tracking)")
    else:
        print(f"  NVML          : not available (install pynvml for GPU utilisation %)")

    # ── Baseline ─────────────────────────────────────────────────────────
    section("Baseline (after imports, before any model load)")
    gc.collect()
    baseline_ram = ram_mb()
    baseline_vram = vram_mb()
    baseline_sys_cpu = system_cpu_percent(interval=0.5)
    print(f"  Process RAM   : {baseline_ram:.1f} MB")
    print(f"  System CPU    : {baseline_sys_cpu:.1f}%")
    print(f"  VRAM          : {fmt_vram(baseline_vram)}")

    # ── Test image ───────────────────────────────────────────────────────
    test_img    = make_test_image()
    test_bytes  = encode_image_to_bytes(test_img)
    img_h, img_w = test_img.shape[:2]
    print(f"\n  Test image    : {img_w}x{img_h} px  ({len(test_bytes)/1024:.1f} KB JPEG)")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "cpu_cores": cpu_count,
            "ram_total_mb": sram['total_mb'],
            "gpu": gpu_name,
            "gpu_vram_mb": gpu_vram_total,
            "cuda_available": CUDA_AVAILABLE,
        },
        "baseline_ram_mb": baseline_ram,
    }

    # ══════════════════════════════════════════════════════════════════════
    #  PART 1: Per-Model Benchmark (default device: GPU if available)
    # ══════════════════════════════════════════════════════════════════════
    section("Part 1: Per-Model Resource Usage (default device)")

    gpu_results = []
    for model_name, model_path in MODELS:
        result = benchmark_single_model(model_name, model_path, test_img, device="", n_runs=5)
        if result:
            gpu_results.append(result)

    all_results["per_model_default"] = gpu_results

    # ══════════════════════════════════════════════════════════════════════
    #  PART 2: Auto-Switching Overhead
    # ══════════════════════════════════════════════════════════════════════
    if (os.path.exists(MODEL_INDOOR_PATH) and os.path.exists(MODEL_OUTDOOR_PATH)):
        switch_results = benchmark_auto_switching(test_img, n_switches=10)
        all_results["auto_switching"] = switch_results
    else:
        print("\n  [SKIP] Auto-switching benchmark: indoor.pt or outdoor.pt not found")

    # ══════════════════════════════════════════════════════════════════════
    #  PART 3: CPU-Only Comparison (only if GPU is available)
    # ══════════════════════════════════════════════════════════════════════
    if CUDA_AVAILABLE:
        section("Part 3: CPU-Only Inference (forced device='cpu')")
        print("  Forcing all models to use CPU to measure the difference vs GPU...\n")

        cpu_results = []
        for model_name, model_path in MODELS:
            result = benchmark_single_model(model_name, model_path, test_img, device="cpu", n_runs=5)
            if result:
                cpu_results.append(result)

        all_results["per_model_cpu_only"] = cpu_results
    else:
        print("\n  [SKIP] CPU vs GPU comparison: no CUDA GPU detected (Part 1 already used CPU)")
        cpu_results = []

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════
    section("Final Report")

    # ── Table 1: Per-model comparison ────────────────────────────────────
    if gpu_results:
        print("\n  ┌─ Per-Model Resource Usage (default device) ────────────────────────────────────────────────────┐")
        header = (f"  │ {'Model':<10} {'File':>6}  {'Load':>6}  {'dRAM':>7}  "
                  f"{'Avg ms':>7}  {'Min ms':>7}  {'Max ms':>7}  "
                  f"{'CPU%':>5}  {'GPU%':>5}  {'VRAM':>8} │")
        print(header)
        print(f"  │ {'─'*10} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*8} │")

        for r in gpu_results:
            vram_str = f"{r['peak_vram_mb']:.0f} MB" if r["peak_vram_mb"] else "n/a"
            gpu_str  = f"{r['avg_gpu_pct']:.0f}" if r["avg_gpu_pct"] is not None else "n/a"
            print(f"  │ {r['name']:<10} {r['file_mb']:>5.1f}M  "
                  f"{r['load_time_s']:>5.2f}s  "
                  f"{r['load_ram_delta_mb']:>+6.1f}M  "
                  f"{r['avg_latency_ms']:>6.1f}ms  "
                  f"{r['min_latency_ms']:>6.1f}ms  "
                  f"{r['max_latency_ms']:>6.1f}ms  "
                  f"{r['avg_cpu_pct']:>4.0f}%  "
                  f"{gpu_str:>4}%  "
                  f"{vram_str:>8} │")
        print(f"  └{'─' * 97}┘")

    # ── Table 2: CPU vs GPU comparison ───────────────────────────────────
    if cpu_results and gpu_results:
        print("\n  ┌─ CPU vs GPU Comparison ─────────────────────────────────────────────────────────────┐")
        print(f"  │ {'Model':<10} {'GPU ms':>8}  {'CPU ms':>8}  "
              f"{'Slowdown':>9}  {'GPU RAM':>8}  {'CPU RAM':>8}  {'ΔRAM':>8} │")
        print(f"  │ {'─'*10} {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*8} │")

        for gr in gpu_results:
            cr = next((c for c in cpu_results if c["name"] == gr["name"]), None)
            if cr is None:
                continue
            slowdown = cr["avg_latency_ms"] / gr["avg_latency_ms"] if gr["avg_latency_ms"] > 0 else 0
            ram_diff = cr["load_ram_delta_mb"] - gr["load_ram_delta_mb"]
            print(f"  │ {gr['name']:<10} "
                  f"{gr['avg_latency_ms']:>7.1f}ms  "
                  f"{cr['avg_latency_ms']:>7.1f}ms  "
                  f"{slowdown:>8.1f}x  "
                  f"{gr['load_ram_delta_mb']:>+7.1f}M  "
                  f"{cr['load_ram_delta_mb']:>+7.1f}M  "
                  f"{ram_diff:>+7.1f}M │")
        print(f"  └{'─' * 89}┘")

        print("\n  Key observations:")
        avg_gpu_lat = sum(r["avg_latency_ms"] for r in gpu_results) / len(gpu_results)
        avg_cpu_lat = sum(r["avg_latency_ms"] for r in cpu_results) / len(cpu_results)
        overall_slowdown = avg_cpu_lat / avg_gpu_lat if avg_gpu_lat > 0 else 0
        print(f"    • CPU inference is ~{overall_slowdown:.1f}x slower on average")
        print(f"    • GPU avg: {avg_gpu_lat:.1f} ms  →  CPU avg: {avg_cpu_lat:.1f} ms")

        gpu_vram_max = max((r["peak_vram_mb"] or 0) for r in gpu_results)
        cpu_extra_ram = sum(cr["load_ram_delta_mb"] - gr["load_ram_delta_mb"]
                           for gr, cr in zip(gpu_results, cpu_results)
                           if any(c["name"] == gr["name"] for c in cpu_results)) / len(gpu_results)
        print(f"    • GPU mode uses {gpu_vram_max:.0f} MB VRAM but less system RAM")
        if cpu_extra_ram > 0:
            print(f"    • CPU mode uses ~{cpu_extra_ram:.0f} MB MORE system RAM (tensors stay on CPU)")
        else:
            print(f"    • CPU mode uses ~{abs(cpu_extra_ram):.0f} MB LESS system RAM")

    # ── Table 3: Auto-switching summary ──────────────────────────────────
    if "auto_switching" in all_results:
        sw = all_results["auto_switching"]
        print(f"\n  ┌─ Auto-Switching Impact ────────────────────────────────────────────┐")
        print(f"  │ Metric                           │ Value                            │")
        print(f"  │ {'─'*34} │ {'─'*32} │")
        print(f"  │ RAM with both models cached       │ {sw['ram_both_cached_mb']:>7.1f} MB                       │")
        print(f"  │ Cached switch (avg latency)        │ {sw['cached_switch_avg_ms']:>7.1f} ms  (no reload needed) │")
        print(f"  │ Cold switch (avg total)            │ {sw['cold_switch_avg_ms']:>7.1f} ms  (gc+reload+infer)  │")
        print(f"  │ ├─ gc.collect + cache clear        │ {sw['cold_gc_avg_ms']:>7.1f} ms                       │")
        print(f"  │ └─ Model reload from disk (I/O)    │ {sw['cold_reload_avg_ms']:>7.1f} ms                       │")
        print(f"  │ Cold vs cached overhead            │ {sw['cold_switch_avg_ms'] - sw['cached_switch_avg_ms']:>+7.1f} ms                       │")
        print(f"  └{'─' * 69}┘")

        print("\n  Analysis:")
        print(f"    • final4.py caches models in _model_cache (dict), so auto-switching")
        print(f"      between indoor↔outdoor costs only ~{sw['cached_switch_avg_ms']:.0f} ms (same as normal inference).")
        print(f"    • If models were evicted (cold switch), the reload I/O penalty is")
        print(f"      ~{sw['cold_reload_avg_ms']:.0f} ms — making it {sw['cold_switch_avg_ms']/sw['cached_switch_avg_ms']:.1f}x slower.")
        print(f"    • Keeping both models cached costs ~{sw['ram_both_cached_mb'] - baseline_ram:.0f} MB of RAM (acceptable for most systems).")
        print(f"    • Recommendation: Keep the default caching strategy. The ~5 MB per")
        print(f"      model is negligible compared to the reload overhead.")

    # ── System requirements ──────────────────────────────────────────────
    print(f"\n  ┌─ Recommended Minimum System Specs ────────────────────────────────┐")

    if gpu_results:
        worst_ram = baseline_ram + sum(r["load_ram_delta_mb"] for r in gpu_results) + 256
        worst_vram = max((r["peak_vram_mb"] for r in gpu_results if r["peak_vram_mb"] is not None), default=None)
        avg_ms = sum(r["avg_latency_ms"] for r in gpu_results) / len(gpu_results)
        fps = 1000 / avg_ms if avg_ms > 0 else 0

        print(f"  │ RAM (all 3 models cached)          │ {worst_ram:>6.0f} MB  ({worst_ram/1024:.1f} GB)        │")
        if worst_vram is not None:
            vram_min = worst_vram + 256
            print(f"  │ GPU VRAM (minimum)                 │ {vram_min:>6.0f} MB  ({vram_min/1024:.1f} GB)        │")
        else:
            print(f"  │ GPU VRAM                           │ Not required (CPU mode)         │")
        print(f"  │ Avg latency (single image)          │ {avg_ms:>6.1f} ms  (~{fps:.1f} fps)          │")
    print(f"  │                                                                     │")
    print(f"  │ Notes:                                                               │")
    print(f"  │   • final4.py caches all 3 models after first use                   │")
    print(f"  │   • Uvicorn adds ~30-50 MB overhead per worker                      │")
    print(f"  │   • Video analysis buffers frames; budget +200 MB                   │")
    print(f"  └{'─' * 69}┘")

    # ── Save results to JSON ─────────────────────────────────────────────
    output_path = os.path.join(BASE_DIR, "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
