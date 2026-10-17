# Task 1: GPU Threads and Data Assignment

English | [简体中文](note.zh.md)

When adding two arrays on a GPU, we specify both the number of blocks and the number of threads per block. All blocks in one kernel launch form a grid. Within each block, threads are organized into wavefronts according to the kernel's wave size. A lane represents a thread's position within its wavefront.

Every thread executes the same kernel but works on its assigned data. In this example, each thread calculates its array index using:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Here, `blockIdx.x` is the block number, `blockDim.x` is the number of threads per block, and `threadIdx.x` is the thread's local number within its block.

If `i` is within the array bounds, the thread reads `A[i]` and `B[i]`, adds them, and writes the result to `C[i]`. Extra threads skip the array access:

```cpp
if (i < n) {
    C[i] = A[i] + B[i];
}
```

Because these additions are independent, different threads can process them in parallel.

![Sketch of threads adding corresponding array elements, with a separate wave32 grouping example](assets/array-addition-sketch.png)

In the main diagram, four threads in one block calculate `[1, 2, 3, 4] + [10, 20, 30, 40] = [11, 22, 33, 44]`. Each thread reads, adds, and writes its own pair of elements. The sidebar is a separate example: a 64-thread block using wave32 contains two wavefronts. The four-thread indexing example does not imply a wave size of four.

**Corrections to my initial understanding:** I specify both the block count and threads per block. The wavefront count follows from the block size and the kernel's wave size. A lane identifies a position within a wavefront; the kernel's index formula determines the array element a thread accesses.

Source: [Hello GPU, Chapter 2: Programming Model and Wavefront Execution](https://datawhalechina.github.io/hello-gpu/part0-intro/chapter2/).
