# Task 4：看懂并手动修改向量加法（HIP 路线）

[English](note.en.md) | 简体中文

实验环境：AMD Radeon Cloud（One-click 资源组），AMD Radeon Pro W7900D（gfx1100，48 个 CU，wave32），HIP 7.15。运行时间为 2026-11-02（UTC）。

## 线程、线程块、下标与越界判断

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
```

- **线程（Thread）：** 每个线程计算一个输出元素。
- **线程块（Block）：** 一组线程。这里每个 Block 有 256 个线程（`blockDim.x = 256`）。
- **下标（Index）：** `blockIdx.x * blockDim.x + threadIdx.x` 给每个线程算出自己负责的数组位置。例如块 2 中的线程 5 处理 `2 × 256 + 5 = 517`。
- **块数：** `blocks = (n + threads - 1) / threads` 向上取整，保证覆盖全部元素。
- **越界判断（Bounds Check）：** `if (idx < n)` 让最后一个块里多出来的线程不读写数组范围之外的位置。

## 修改数组长度

我把 `vector_add.hip` 中的 `const int n = 1 << 20;` 改为 `const int n = 1000003;`。这个长度不能被 256 整除。

| 长度 | 块数 | 启动线程数 | 多余线程 | 结果 |
|---:|---:|---:|---:|---|
| 1,048,576（原始） | 4096 | 1,048,576 | 0 | `max_error: 0`，`status: PASS` |
| 1,000,003（修改后） | 3907 | 1,000,192 | 189 | `max_error: 0`，`status: PASS` |

长度为 1,000,003 时，前 3906 个完整的块覆盖 999,936 个元素。最后一个块仍然启动 256 个线程，但只有 67 个线程的下标有效，其余 189 个线程因为 `if (idx < n)` 跳过计算。

我还运行了第 8 章的 HIP 程序（`vector_add_hip.hip`，版本 `v0`，block 256，warmup 10，repeat 50），两个长度都通过了检查：

| 长度 | Grid | 正确性 | 中位数 | 有效带宽 |
|---:|---:|---|---:|---:|
| 16,777,216 | 65,536 | OK | 0.2919 ms | 690 GB/s |
| 1,000,003 | 3,907 | OK | 0.0221 ms | 543 GB/s |

## 为什么需要越界判断

块数是向上取整的，所以最后一个块的线程数可能多于剩下的元素数。如果没有 `if (idx < n)`，多出来的线程就会读写数组末尾之后的内存。那部分内存属于别的数据，可能导致结果错误或程序崩溃。

来源：[Hello GPU 第 4 章](https://datawhalechina.github.io/hello-gpu/part0-intro/chapter4/)；[第 8 章：逐元素算子](https://datawhalechina.github.io/hello-gpu/part2-kernels/chapter8/)。
