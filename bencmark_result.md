# Benchmark Results: final4.py Resource Usage

**System**: RTX 2050 (4 GB VRAM) · 12 cores · 16 GB RAM · Python 3.10 · PyTorch 2.12.1+cu132

---

## Part 1: Per-Model Resource Usage (GPU)

| Model | File | Load Time | RAM Δ | Avg Latency | Min | Max | CPU% | GPU% | Peak VRAM |
|-------|------|-----------|-------|-------------|-----|-----|------|------|-----------|
| **indoor** | 5.1 MB | 0.18s | +113.1 MB | **14.4 ms** | 11.9 ms | 16.2 ms | 81% | 1% | 52 MB |
| **outdoor** | 5.1 MB | 0.07s | +1.3 MB | **12.3 ms** | 11.2 ms | 14.7 ms | 206% | 11% | 52 MB |
| **base** | 5.3 MB | 0.10s | +0.8 MB | **15.2 ms** | 12.9 ms | 17.6 ms | 79% | 11% | 59 MB |

> [!NOTE]
> - The indoor model's first-time RAM spike (+113 MB) is from initializing CUDA runtime and PyTorch's internal buffers (one-time cost at first model load). Subsequent models reuse this memory.
> - CPU% > 100% indicates multi-core utilization (e.g., 206% ≈ 2 cores busy during pre/post-processing).
> - All three models use **~52–59 MB VRAM** for inference — extremely lightweight for a 4 GB GPU.

---

## Part 2: Auto-Switching Overhead

### Scenario A: Cached switching (final4.py default behavior)

`final4.py` caches loaded models in `_model_cache` dict. When switching between indoor↔outdoor, **no reload happens** — just a pointer swap.

| Switch | Latency | RAM Δ |
|--------|---------|-------|
| Avg across 10 switches | **13.0 ms** | **+0.0 MB** |

> [!TIP]
> **Zero overhead.** Cached switching costs the same as a normal inference (~13 ms). No gc, no I/O, no memory churn.

### Scenario B: Cold switching (if cache were cleared)

What if we evicted models and forced reload from disk each time?

| Metric | Value |
|--------|-------|
| gc.collect + torch.cuda.empty_cache | **79.5 ms** |
| Model reload from disk (I/O) | **58.1 ms** |
| + Inference | ~67 ms |
| **Total cold switch** | **205.5 ms** |
| **Overhead vs cached** | **+192.5 ms (15.8x slower)** |

> [!IMPORTANT]
> **Recommendation**: Keep the default caching strategy. Both models cached costs only ~10 MB of model weight RAM combined. The reload I/O penalty (58 ms) + gc overhead (80 ms) makes cold switching **15.8x slower** — completely unnecessary for these small models.

### RAM with both models cached

| State | Process RAM |
|-------|-------------|
| Baseline (imports only) | 543 MB |
| Both indoor + outdoor cached | 1,429 MB |
| Difference | ~886 MB |

> [!NOTE]
> The ~886 MB delta includes PyTorch CUDA runtime, ultralytics overhead, and model weights. The actual model weights are only ~5 MB each — the rest is framework overhead that exists regardless of caching strategy.

---

## Part 3: CPU-Only vs GPU Comparison

| Model | GPU Latency | CPU Latency | Slowdown | GPU RAM Δ | CPU RAM Δ |
|-------|-------------|-------------|----------|-----------|-----------|
| **indoor** | 14.4 ms | 40.7 ms | **2.8x** | +113.1 MB | +0.0 MB |
| **outdoor** | 12.3 ms | 40.4 ms | **3.3x** | +1.3 MB | +5.1 MB |
| **base** | 15.2 ms | 54.1 ms | **3.5x** | +0.8 MB | +3.7 MB |

### Key findings

```
GPU average: 14.0 ms  (~71.6 fps theoretical)
CPU average: 45.1 ms  (~22.2 fps theoretical)
CPU is ~3.2x slower on average
```

- **CPU mode uses ~800% CPU** (saturates multiple cores via OpenMP/MKL threading)
- **GPU mode uses only ~1–11% GPU** — the models are so small the GPU barely breaks a sweat
- **GPU mode uses 52–59 MB VRAM** — trivial for any modern GPU
- **CPU mode uses slightly less system RAM** since tensors stay on CPU (no CUDA runtime overhead)

> [!IMPORTANT]
> For a **CPU-only deployment** (no GPU machine), expect:
> - ~40–54 ms per image (still real-time at ~20 fps)
> - Higher CPU load (~700–850% across cores)
> - No VRAM needed, slightly less total RAM
> - This is perfectly viable for single-user deployments

---

## Output Files

- [benchmark_memory.py](file:///d:/Code/Python/brin/benchmark_memory.py) — the benchmark script
- [benchmark_results.json](file:///d:/Code/Python/brin/benchmark_results.json) — machine-readable results
