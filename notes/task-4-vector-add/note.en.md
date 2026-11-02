# Task 4: Reading and Changing Vector Add (HIP Route)

English | [简体中文](note.zh.md)

Environment: AMD Radeon Cloud (One-click pool), AMD Radeon Pro W7900D (gfx1100, 48 CUs, wave32), HIP 7.15. Runs on 2026-11-02 (UTC).

## Thread, block, index, and bounds check

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
```

- **Thread:** each thread computes one output element.
- **Block:** a group of threads. Here a block has 256 threads (`blockDim.x = 256`).
- **Index:** `blockIdx.x * blockDim.x + threadIdx.x` gives each thread its own array position. For example, thread 5 in block 2 handles `2 × 256 + 5 = 517`.
- **Block count:** `blocks = (n + threads - 1) / threads` rounds up, so every element is covered.
- **Bounds check:** `if (idx < n)` stops the extra threads in the last block from reading or writing outside the arrays.

## Changing the array length

I changed `const int n = 1 << 20;` to `const int n = 1000003;` in `vector_add.hip`. This length is not a multiple of 256.

| Length | Blocks | Threads launched | Extra threads | Result |
|---:|---:|---:|---:|---|
| 1,048,576 (original) | 4096 | 1,048,576 | 0 | `max_error: 0`, `status: PASS` |
| 1,000,003 (modified) | 3907 | 1,000,192 | 189 | `max_error: 0`, `status: PASS` |

With 1,000,003 elements, 3906 full blocks cover 999,936 elements. The last block still launches 256 threads, but only 67 of them have valid indices. The other 189 threads skip the work because of `if (idx < n)`.

I also ran the Chapter 8 HIP program (`vector_add_hip.hip`, version `v0`, block 256, warmup 10, repeat 50). It passed all checks at both lengths:

| Length | Grid | Correctness | Median | Effective bandwidth |
|---:|---:|---|---:|---:|
| 16,777,216 | 65,536 | OK | 0.2919 ms | 690 GB/s |
| 1,000,003 | 3,907 | OK | 0.0221 ms | 543 GB/s |

## Why the bounds check is needed

The block count is rounded up, so the last block can have more threads than there are remaining elements. Without `if (idx < n)`, those extra threads would read and write past the end of the arrays. That memory belongs to something else, so the result can be wrong or the program can crash.

Sources: [Hello GPU, Chapter 4](https://datawhalechina.github.io/hello-gpu/part0-intro/chapter4/); [Chapter 8: Element-Wise Operators](https://datawhalechina.github.io/hello-gpu/part2-kernels/chapter8/).
