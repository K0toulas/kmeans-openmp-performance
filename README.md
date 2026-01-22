# K-means (C/OpenMP) — VTune-driven Optimization

OpenMP optimization of a K-means clustering implementation (Intel `icx`, `-qopenmp`) with profiling-driven changes and correctness validation.

## Result
- **3.480s (serial)** → **0.189s (OpenMP, 28 threads)** = **18.4× speedup**
- 56 threads: **0.216s (16.1×)** (regression from synchronization/overhead)

## Approach (summary)
- Hotspot identification with **Intel VTune** (distance / nearest-cluster / centroid update).
- Avoided fine-grain and nested parallelism; parallelized over objects (better granularity).
- Used **thread-local accumulators** for centroid sums/counts; minimized synchronization and tuned scheduling (`dynamic,25`).
- Verified outputs vs baseline (membership identical; centers ~`1e-6` abs diff from FP reduction order).

## Build
Requires Intel oneAPI compiler (`icx`):
```bash
make -C src
