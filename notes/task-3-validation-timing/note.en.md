# Task 3: Running the Program, Checking Correctness, and Timing

English | [简体中文](note.zh.md)

Environment: AMD Radeon Cloud, AMD Radeon Pro W7900D (gfx1100), AMD EPYC 9334 CPU, HIP 7.15, PyTorch 2.13.0+rocm10.0.0. Runs on 2026-10-27 (UTC).

## Correctness

`vector_add.hip` is a hand-written HIP vector add. The host fills `a` with 1 and `b` with 2, copies them to the GPU, launches the kernel, waits for it to finish, copies `c` back, and checks that every element equals 3.

```bash
hipcc vector_add.hip -O2 -o vector_add
./vector_add
```

```text
device_name: AMD Radeon Pro W7900D
vector_size: 1048576
blocks: 4096
threads_per_block: 256
max_error: 0
status: PASS
```

`max_error: 0` and `status: PASS` mean the result is correct. This program checks the result only; it does not time the HIP kernel.

## Timing

`benchmark_vector_add.py` times **PyTorch's** `torch.add(a, b, out=c)`, not `vector_add.hip`. Both do the same math, but they are different implementations.

```bash
python benchmark_vector_add.py --size 16777216 --warmup 5 --repeat 30 --cpu-threads 1
```

All runs use `float32`, warmup 5, repeat 30, and 1 CPU thread. Both CPU and GPU validation passed at every size.

| Size (elements) | GPU median | GPU mean | GPU min | GPU bandwidth (median) | CPU median |
|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 0.0323 ms | 0.0314 ms | 0.0238 ms | 389 GB/s | 0.131 ms |
| 16,777,216 | 0.292 ms | 0.303 ms | 0.291 ms | 689 GB/s | 4.58 ms |
| 67,108,864 | 1.117 ms | 1.118 ms | 1.115 ms | 721 GB/s | 19.37 ms |

## What the timing measures

- The GPU time covers only one `torch.add` call. A GPU event is recorded before and after the add on the same stream. The host waits for the end event with `end.synchronize()` and then reads the time between the two events.
- GPU work is asynchronous: the CPU can move on before the GPU finishes. A plain CPU clock around the call could measure only the submission, so the GPU path uses events.
- Warmup (5 runs) reduces first-run effects. Repeating 30 times shows how much the samples vary. The median is the main value.
- Not included: creating the inputs, allocating the output, host–device copies, and the correctness check.
- The CPU time is a wall clock around the same call on the CPU. It includes the Python call overhead. The CPU/GPU ratio is not the speedup of a whole program.

## Effective bandwidth

Each element reads `a[i]` and `b[i]` and writes `c[i]`. That is 3 × 4 bytes = 12 bytes per element. For N = 16,777,216:

12 × 16,777,216 = 201,326,592 bytes; 201,326,592 / 0.292320 ms ≈ 689 GB/s.

This is the data the algorithm needs divided by the time. It is not the measured physical memory traffic.

## Observations

- For the larger sizes, the GPU time grows roughly with N, and the effective bandwidth stays near 690–720 GB/s. Vector add does one add per 12 bytes, so moving data dominates.
- At 1M elements, the bandwidth is much lower, and min (0.024 ms) is far from median (0.032 ms). The add itself is very short, so launch and submission costs take a larger share and make samples less stable.

## Baseline

The original `vector_add.hip`, `benchmark_vector_add.py`, and the run logs are saved in `/workspace/task3-baseline` in the cloud workspace. They are the baseline for the Task 5 comparison.

Sources: [Hello GPU, Chapter 4: First Program and Performance Analysis](https://datawhalechina.github.io/hello-gpu/part0-intro/chapter4/); [Chapter 5: Benchmarks and Reliable Timing](https://datawhalechina.github.io/hello-gpu/part1-profiling/chapter5/).
