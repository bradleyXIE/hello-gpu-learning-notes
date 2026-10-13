# Task 0：环境准备

[English](note.en.md) | 简体中文

## PyTorch 与 HIP

PyTorch 提供加法等算子的实现。HIP 让我自己编写 GPU kernel。Kernel 是由 GPU 线程执行的函数。

在本次实验中，PyTorch 管理 Tensor，我写的 HIP kernel 则定义 GPU 如何计算每个输出元素。

## 编译缓存

Notebook 辅助工具会缓存编译后的 kernel。当源码和编译配置一致时，后续调用可以复用它。输入和输出 Tensor 与编译缓存是分开的。

## 线程块

线程块（thread block）是一组 GPU 线程。在这个加法 kernel 中，每个线程计算一个输出元素。在其他 kernel 中，一个线程可以处理多个元素，或执行多个计算步骤。

## 实验结果

HIP 编译缓存已复用。513 个元素的加法通过了完整数组检查及最后一个元素检查（下标 512）。

![HIP 缓存与边界检查成功截图](assets/hip-cache-and-boundary-check.png)
