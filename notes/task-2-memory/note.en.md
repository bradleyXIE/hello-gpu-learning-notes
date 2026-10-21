# Task 2: Where GPU Data Is Stored

English | [简体中文](note.zh.md)

In vector add, the input arrays `A` and `B` and the output array `C` live in **global memory**. On a discrete GPU, global memory is backed by the card's VRAM. "Global memory" describes the space the program accesses; "VRAM" describes the hardware that holds the data.

Each thread handles its own element `i`. The flow has three steps:

1. **Read:** the thread loads `A[i]` and `B[i]` from global memory into its own **registers**. Reading copies the values, so `A[i]` and `B[i]` keep their original values. A read takes time: the request goes out first, and the data arrives later.
2. **Add:** the **execution unit** uses the values in the registers to compute the sum. The result is first kept in a register. Registers store values; execution units compute.
3. **Write back:** the thread writes the result to `C[i]` in global memory. `C[i]` changes only after this step.

For example, for `C[2] = A[2] + B[2]`, the thread reads 3 and 30, adds them to get 33, and writes 33 to `C[2]`.

All lanes of a wavefront issue their reads together, and each lane supplies its own address. When neighbouring lanes read neighbouring elements, the hardware can combine the requests. This is called **memory coalescing**. A **cache** may already hold a copy of the needed data. Coalescing depends on how the addresses are arranged; a cache hit depends on whether a copy already exists.

**LDS** (Local Data Share, `__shared__` in HIP) is a space that threads in the same block use to exchange results. One example is a sum, where threads must combine partial results. In that case, threads write to LDS, wait at `__syncthreads()`, and then read. Vector add does not need LDS, because each thread's element is independent.

**Corrections to my initial understanding:** I first thought the data is placed in registers and every lane in the wavefront can access it, and that results are written to LDS. In fact, the arrays stay in global memory. Each thread reads only its own inputs into its own registers, so lanes do not read each other's registers. The result is written back to `C[i]` in global memory, not to LDS.

Source: [Hello GPU, Chapter 3: Data Storage and Access](https://datawhalechina.github.io/hello-gpu/part0-intro/chapter3/).
