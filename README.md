# OpenMP K-means Performance Optimization 

Measurement driven optimization of K-means clustering using Intel VTune and OpenMP.

## Highlights
- Best runtime: **3.480s (serial)** → **0.189s (OpenMP @ 28 threads)** = **18.4× speedup**
- 56 threads: **0.216s (16.1×)** (regression due to synchronization/overhead)
- Correctness: membership identical; cluster centers differ by ~**1e-6** (floating-point reduction order)

## Build
```bash
make -C src
