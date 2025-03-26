#!/usr/bin/env python3
import torch
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Import your custom CUDA kernel implementation of ReLU.
# Ensure your relu_custom function accepts a CUDA tensor as input and returns a CUDA tensor.
from my_activations import relu_custom

def benchmark_kernel(kernel_func, x, num_iters=100):
    # Warm-up iterations to ensure kernels are loaded and any lazy initialization is done.
    for _ in range(10):
        kernel_func(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        # Run the kernel.
        kernel_func(x)
    torch.cuda.synchronize()
    end = time.time()
    avg_time = (end - start) / num_iters
    return avg_time

def run_benchmark(sizes, num_iters=100):
    print("Benchmarking custom CUDA ReLU vs PyTorch ReLU")
    print("------------------------------------------------")

    # Lists for collecting results for plotting.
    num_elements = []
    custom_times = []
    pytorch_times = []

    for size in sizes:
        # Create a random tensor of the given size on the CUDA device.
        x = torch.randn(*size, device='cuda')
        total_elements = np.prod(size)
        print(f"Input size: {size} (total elements: {total_elements})")

        # Benchmark the custom CUDA ReLU kernel.
        custom_time = benchmark_kernel(relu_custom, x, num_iters) * 1e3  # convert to ms
        
        # Benchmark the PyTorch built-in ReLU.
        pytorch_time = benchmark_kernel(torch.nn.functional.relu, x, num_iters) * 1e3  # convert to ms
        
        print(f"Custom CUDA ReLU: {custom_time:.3f} ms per iteration")
        print(f"PyTorch ReLU:     {pytorch_time:.3f} ms per iteration")
        print("------------------------------------------------")

        num_elements.append(total_elements)
        custom_times.append(custom_time)
        pytorch_times.append(pytorch_time)
    
    return num_elements, custom_times, pytorch_times

def plot_results(num_elements, custom_times, pytorch_times, filename="benchmark_results.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(num_elements, custom_times, marker='o', label='Custom CUDA ReLU')
    plt.plot(num_elements, pytorch_times, marker='o', label='PyTorch ReLU')
    plt.xlabel('Number of elements in input tensor')
    plt.ylabel('Average execution time (ms)')
    plt.title('ReLU Kernel Benchmark')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Benchmark results plot saved to {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark custom CUDA ReLU kernel vs PyTorch built-in ReLU."
    )
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations per test")
    parser.add_argument("--plot", type=str, default="benchmark_results.png", help="Filename to save plot")
    args = parser.parse_args()

    # List of input sizes to test (as tuples representing tensor dimensions).
    sizes = [
        (1024,),              # 1D tensor
        (2048,),              # Larger 1D tensor
        (1024, 1024),         # 2D tensor
        (2048, 2048),         # Larger 2D tensor
        (4096, 4096),         # Very large 2D tensor
        (512, 512, 512),      # 3D tensor
        (1024, 1024, 128),    # Large 3D variation
        (128, 128, 128, 128), # 4D tensor
        (8192, 4096),         # Extremely large 2D tensor
    ]

    num_elements, custom_times, pytorch_times = run_benchmark(sizes, num_iters=args.iters)
    plot_results(num_elements, custom_times, pytorch_times, filename=args.plot)