### ## Accelerating ResNet-50 Inference with TensorRT

This project demonstrates an end-to-end workflow for optimizing a PyTorch ResNet-50 model by compiling it into a hardware-specific inference engine using TensorRT. It explores systems-level optimization techniques like kernel fusion and INT8 quantization to dramatically reduce latency and memory footprint by leveraging specialized GPU hardware features like Tensor Cores.

### ## The Problem: The General-Purpose Gap

A baseline PyTorch model can be considered "slow" for production inference because it is **hardware-agnostic**. It's designed for flexibility, not peak performance on a specific GPU. Its operations (convolutions, activations) are executed as a series of separate, generic CUDA kernel launches, leading to significant memory bandwidth consumption and kernel launch overhead.

### ## The Solution: A Systems-Aware Compiler

TensorRT acts as a specialized compiler that bridges this gap. It analyzes the entire computation graph and applies several systems-level optimizations:

* **Layer & Tensor Fusion**: TensorRT fuses sequential operations (like a `Convolution -> Bias Add -> ReLU` sequence) into a single, custom CUDA kernel. This classic optimization **reduces memory I/O**, as data doesn't have to be written back to global VRAM between layers, and **minimizes kernel launch overhead**.

* **INT8 Quantization**: By reducing the precision of model weights and activations from FP32 to INT8, we can leverage the **specialized hardware (Tensor Cores)** on modern NVIDIA GPUs. These cores execute INT8 matrix math orders of magnitude faster than standard FP32 CUDA cores.

* **Kernel Auto-Tuning**: When building the engine, TensorRT profiles multiple implementations of a given operation on the **target GPU** (like the T4 in the Colab environment) and selects the fastest one, ensuring a hardware-specific, optimal configuration.

### ## Future Work

* **Deeper Profiling**: Use NVIDIA's Nsight Systems to profile the baseline and optimized models. This would visually confirm the reduction in kernel launches and reveal exactly how GPU resources (like Tensor Cores) are utilized before and after TensorRT optimization.

* **Custom Layer Integration**: Implement a simple custom layer (e.g., a specialized activation function) with a CUDA kernel and integrate it into the TensorRT graph using the Plugin API. This would demonstrate direct, low-level control over the GPU within a high-level AI framework.
