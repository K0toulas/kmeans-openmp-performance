# K-means (C/OpenMP) — VTune-driven Optimization

OpenMP optimization of a K-means clustering implementation (Intel `icx`, `-qopenmp`) with profiling-driven changes and correctness validation.

## Result
- **3.480s (serial)** → **0.189s (OpenMP, 28 threads)** = **18.4× speedup**
- 56 threads: **0.216s (16.1×)** (regression from synchronization/overhead)

## Project summary
Optimized a sequential K-means clustering implementation in C using **OpenMP** on multi-core CPUs. Used **Intel VTune** to identify bottlenecks, implemented parallel regions with reduced synchronization (thread-local accumulation + merge), tuned scheduling (`dynamic,25`), and evaluated scaling across multiple thread counts.
## Approach (summary)
- Hotspot identification with **Intel VTune** (distance / nearest-cluster / centroid update).
- Avoided fine grain and nested parallelism, parallelized over objects (better granularity).
- Used **thread-local accumulators** for centroid sums/counts, minimized synchronization and tuned scheduling (`dynamic,25`).
- Verified outputs vs baseline (membership identical; centers ~`1e-6` abs diff from FP reduction order).

## Build
Requires Intel oneAPI compiler (`icx`):
```bash
make -C src
