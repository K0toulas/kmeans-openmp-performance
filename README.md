# K-means Clustering (C + OpenMP) — Performance Engineering

Performance optimized a sequential K-means clustering implementation in C using **OpenMP** on a multi-core CPU. The focus is on profiling driven optimization, reducing synchronization/critical-path overhead and evaluating scaling behavior across thread counts.

## Project summary
Starting from a correct sequential K-means implementation, I used **Intel VTune** to identify hotspots and then implemented and tuned OpenMP parallel regions. Key changes included handling of per-thread work, minimizing shared-state contention and selecting an effective scheduling strategy for the workload.

## Optimization highlights
- **Profiling-driven workflow:** used Intel VTune to locate hotspots and verify that changes moved time out of the critical path.
- **OpenMP parallelization:** parallelized the compute heavy regions and tuned thread-level behavior.
- **Reduced synchronization overhead:** limited contention by avoiding unnecessary shared updates in hot loops.
- **Scheduling & scaling:** evaluated scaling across thread counts and tuned scheduling (`dynamic,25`) to improve load balance.

## Results
- Reduced runtime from **3.480 s** (serial baseline) to **0.189 s** at **28 threads** (**18.4× speedup**).
- Performance was evaluated across thread counts, best time occurred at 28 threads (56 threads showed diminishing returns due to overhead/pressure).
- 
## Build
This project uses Intel oneAPI `icx` with OpenMP enabled (per `src/Makefile`):
```bash
make -C src
```
## Run 
```bash
./src/seq_main -q -b -n 4 -i Image_data/color17695.bin
./src/seq_main -q    -n 4 -i Image_data/color100.txt
```
## Clean
```bash
make -C src clean
```
## Scaling  
Control threads via:  
 `OMP_NUM_THREADS=<N>`



