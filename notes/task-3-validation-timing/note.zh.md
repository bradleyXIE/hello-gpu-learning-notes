# Task 3：跑通程序、检查正确性与计时

[English](note.en.md) | 简体中文

实验环境：AMD Radeon Cloud，AMD Radeon Pro W7900D（gfx1100），AMD EPYC 9334 CPU，HIP 7.15，PyTorch 2.13.0+rocm10.0.0。运行时间为 2026-10-27（UTC）。

## 正确性检查

`vector_add.hip` 是手写的 HIP 向量加法。主机端把 `a` 全部设为 1、`b` 全部设为 2，复制到 GPU，启动 kernel，等待计算完成，再把 `c` 复制回主机，检查每个元素是否等于 3。

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

`max_error: 0` 和 `status: PASS` 表示结果正确。这个程序只检查结果，没有对 HIP kernel 计时。

## 计时

`benchmark_vector_add.py` 测量的是 **PyTorch** 的 `torch.add(a, b, out=c)`，不是 `vector_add.hip`。两者做相同的数学运算，但属于不同的实现。

```bash
python benchmark_vector_add.py --size 16777216 --warmup 5 --repeat 30 --cpu-threads 1
```

所有运行都使用 `float32`、预热（warmup）5 次、重复（repeat）30 次、1 个 CPU 线程。每个规模下，CPU 和 GPU 的结果检查都通过了。

| 规模（元素数） | GPU 中位数 | GPU 平均值 | GPU 最小值 | GPU 有效带宽（中位数） | CPU 中位数 |
|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 0.0323 ms | 0.0314 ms | 0.0238 ms | 389 GB/s | 0.131 ms |
| 16,777,216 | 0.292 ms | 0.303 ms | 0.291 ms | 689 GB/s | 4.58 ms |
| 67,108,864 | 1.117 ms | 1.118 ms | 1.115 ms | 721 GB/s | 19.37 ms |

## 计时测量了哪些操作

- GPU 时间只包含一次 `torch.add` 调用。在同一条执行流（stream）上，加法前后各记录一个 GPU event；主机用 `end.synchronize()` 等待结束 event 完成，再读取两个 event 之间的时间差。
- GPU 执行是异步的（asynchronous）：GPU 还没算完，CPU 就可能继续往下执行。直接用 CPU 时钟计时可能只量到提交过程，所以 GPU 路径使用 event 计时。
- 预热 5 次可以减轻首次运行的影响；重复 30 次可以观察样本的波动。主要看中位数（median）。
- 不在计时范围内的操作：创建输入、分配输出、主机与设备之间的拷贝，以及正确性检查。
- CPU 时间是在 CPU 上对同一调用前后读取的墙钟时间（wall clock），包含 Python 调用开销。CPU 与 GPU 的时间比不等于整个程序的加速比。

## 有效带宽

每个元素读取 `a[i]` 和 `b[i]`，写入 `c[i]`，共 3 × 4 = 12 字节。N = 16,777,216 时：

12 × 16,777,216 = 201,326,592 字节；201,326,592 / 0.292320 ms ≈ 689 GB/s。

这是算法需要处理的数据量除以时间，不是实测的物理显存流量。

## 观察

- 规模较大时，GPU 时间大致随 N 线性增长，有效带宽稳定在 690–720 GB/s 左右。向量加法每 12 字节只做一次加法，主要开销在数据搬运上。
- 规模为 1M 时，有效带宽明显偏低，最小值（0.024 ms）和中位数（0.032 ms）相差较大。这时加法本身很短，启动和提交的开销占比变大，样本也更不稳定。

## 基线

原始的 `vector_add.hip`、`benchmark_vector_add.py` 和运行日志保存在云端工作区的 `/workspace/task3-baseline` 中，作为 Task 5 对比的基线。

来源：[Hello GPU 第 4 章：第一个程序 + 性能分析](https://datawhalechina.github.io/hello-gpu/part0-intro/chapter4/)；[第 5 章：benchmark 与可信计时](https://datawhalechina.github.io/hello-gpu/part1-profiling/chapter5/)。
