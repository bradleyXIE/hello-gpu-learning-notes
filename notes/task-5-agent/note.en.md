# Task 5: Agent-Assisted Optimization of Vector Add

English | [简体中文](note.zh.md)

Environment: AMD Radeon Cloud (One-click pool), AMD Radeon Pro W7900D (gfx1100, 48 CUs, wave32), HIP 7.15. Runs on 2026-11-02 (UTC). Agent: Claude Code (Claude Opus 5.5).

## What I gave the Agent

- **GPU:** AMD Radeon Pro W7900D, gfx1100.
- **Code:** the Chapter 8 HIP vector add, `vector_add_v0` (one thread per element, default block 256).
- **Correctness check:** the program compares the output with a CPU reference before and after timing (`correct`, `precheck` and `postcheck` must all be `OK`).
- **Timing:** GPU events, warmup 10, repeat 50, N = 16,777,216 `float32`, median of each process.

Key prompt: "Explain this vector-add kernel. Suggest one simple optimization, such as changing the block size. Keep the original version, make one change, check correctness first, then time both versions with the same input size and timing method."

## The Agent's explanation and suggestion

Vector add does one addition for every 12 bytes moved, so its speed is limited by memory bandwidth, not by computation. The Agent suggested trying different thread block sizes. Then it changed the program's default block size from 256 to 512 (one line, `int block = 512;`) in a copy of the file. The original file was kept unchanged.

## Block-size scan

Before choosing 512, the Agent measured five block sizes. The block sizes were interleaved across 3 rounds to reduce the effect of run order. All runs passed the correctness checks.

| Block | Grid | Median of runs |
|---:|---:|---:|
| 64 | 262,144 | 0.2952 ms |
| 128 | 131,072 | 0.2816 ms |
| 256 (original) | 65,536 | 0.2808 ms |
| 512 | 32,768 | 0.2794 ms |
| 1024 | 16,384 | 0.2951 ms |

The first, non-interleaved scan was misleading. The first process of each block size was about 0.295 ms, and the next ones about 0.280 ms. Changing the run order is what showed this.

## Before vs. after

Both versions were run in alternation, 5 independent processes each, with the same input and timing.

| Version | Correctness | Median of 5 processes | Range |
|---|---|---:|---:|
| Original (block 256) | OK ×5 | 0.2810 ms | 0.2802–0.2958 ms |
| Modified (block 512) | OK ×5 | 0.2802 ms | 0.2791–0.2956 ms |

At the odd length 1,000,003, both versions also passed the checks (0.0208 ms vs. 0.0213 ms).

## Summary

- **What the Agent changed:** the default thread block size, from 256 to 512.
- **Correct?** Yes, every run passed.
- **Faster?** No meaningful speedup. The difference is about 0.3%, while single processes for both versions jump between about 0.280 ms and 0.295 ms. The noise is larger than the change.
- **What I learned:** for a memory-bound kernel like vector add, block size between 128 and 512 barely matters. Very small (64) or very large (1024) blocks were about 5% slower in this scan. A fair comparison needs the same input, the same timing method, several independent runs, and alternating run order.

Sources: [Hello GPU, Chapter 8](https://datawhalechina.github.io/hello-gpu/part2-kernels/chapter8/); [Part 3: Kernel-Level Agent](https://datawhalechina.github.io/hello-gpu/part3-agent/chapter14/).
