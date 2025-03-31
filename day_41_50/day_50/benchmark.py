import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from torch.utils.cpp_extension import load, include_paths as torch_include_paths

nvcc_compile_args = [
    "-O3",
    "--ftz=true",
    "-Xptxas=-O3",
    "--maxrregcount=64",
    "--default-stream=per-thread",
    "--restrict",
    "-lineinfo",
    *[f"-I{path}" for path in torch_include_paths()],
]
# Import the C++ extension
# Note: This assumes the above C++ code is compiled as a module named "cuda_embedding"


# Just-in-time compile the extension
# You can skip this if you've already built the extension with setup.py
custom_embedding = load(
    name="cuda_embedding",
    sources=["emb.cu"],
    verbose=True,
    extra_cuda_cflags=nvcc_compile_args,
    with_cuda=True,
)


def benchmark_embedding(vocab_size, embedding_dim, batch_sizes, num_iterations=100):
    """
    Benchmark the performance of custom CUDA embedding vs PyTorch embedding.

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of each embedding
        batch_sizes: List of batch sizes to test
        num_iterations: Number of iterations for each test

    Returns:
        Dict containing forward and backward timing results
    """
    # Initialize results dictionary
    results = {
        "batch_sizes": batch_sizes,
        "pytorch_forward": [],
        "pytorch_backward": [],
        "custom_forward": [],
        "custom_backward": [],
    }

    # Create the embeddings
    torch_embedding = nn.Embedding(vocab_size, embedding_dim).cuda()
    cuda_embedding = custom_embedding.CudaEmbedding(vocab_size, embedding_dim)

    # Copy weights to ensure fair comparison
    with torch.no_grad():
        cuda_embedding.weight.copy_(torch_embedding.weight)

    # Dummy loss function
    criterion = nn.MSELoss()

    # Run benchmarks for each batch size
    for batch_size in batch_sizes:
        print(f"Benchmarking batch size: {batch_size}")

        # Create random indices
        indices = torch.randint(0, vocab_size, (batch_size,), device="cuda").to(
            torch.int32
        )

        # Create target for backward pass
        target = torch.randn(batch_size, embedding_dim, device="cuda")

        # Benchmark PyTorch embedding
        pytorch_forward_times = []
        pytorch_backward_times = []

        for _ in range(num_iterations):
            # Forward pass
            torch.cuda.synchronize()
            start_time = time.time()

            output = torch_embedding(indices)

            torch.cuda.synchronize()
            forward_time = time.time() - start_time
            pytorch_forward_times.append(forward_time)

            # Backward pass
            loss = criterion(output, target)

            torch.cuda.synchronize()
            start_time = time.time()

            loss.backward()

            torch.cuda.synchronize()
            backward_time = time.time() - start_time
            pytorch_backward_times.append(backward_time)

            # Zero gradients
            torch_embedding.zero_grad()

        # Benchmark Custom CUDA embedding
        custom_forward_times = []
        custom_backward_times = []

        for _ in range(num_iterations):
            # Forward pass
            torch.cuda.synchronize()
            start_time = time.time()

            output = cuda_embedding(indices)

            torch.cuda.synchronize()
            forward_time = time.time() - start_time
            custom_forward_times.append(forward_time)

            # Backward pass
            loss = criterion(output, target)

            torch.cuda.synchronize()
            start_time = time.time()

            loss.backward()

            torch.cuda.synchronize()
            backward_time = time.time() - start_time
            custom_backward_times.append(backward_time)

            # Zero gradients
            cuda_embedding.zero_grad()

        # Calculate mean times after removing outliers
        results["pytorch_forward"].append(np.mean(pytorch_forward_times[10:]))
        results["pytorch_backward"].append(np.mean(pytorch_backward_times[10:]))
        results["custom_forward"].append(np.mean(custom_forward_times[10:]))
        results["custom_backward"].append(np.mean(custom_backward_times[10:]))

    return results


def plot_results(results):
    """Plot the benchmark results"""
    batch_sizes = results["batch_sizes"]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Forward pass plot
    ax1.plot(batch_sizes, results["pytorch_forward"], "b-o", label="PyTorch")
    ax1.plot(batch_sizes, results["custom_forward"], "r-o", label="Custom CUDA")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Forward Pass Time")
    ax1.legend()
    ax1.grid(True)

    # Backward pass plot
    ax2.plot(batch_sizes, results["pytorch_backward"], "b-o", label="PyTorch")
    ax2.plot(batch_sizes, results["custom_backward"], "r-o", label="Custom CUDA")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Backward Pass Time")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("embedding_benchmark.png")
    plt.show()

    # Calculate speedups
    forward_speedup = [
        p / c for p, c in zip(results["pytorch_forward"], results["custom_forward"])
    ]
    backward_speedup = [
        p / c for p, c in zip(results["pytorch_backward"], results["custom_backward"])
    ]

    print("\nSpeedup Summary (PyTorch time / Custom time):")
    print("Batch Size | Forward Speedup | Backward Speedup")
    print("-" * 50)
    for i, bs in enumerate(batch_sizes):
        print(f"{bs:10d} | {forward_speedup[i]:14.2f}x | {backward_speedup[i]:15.2f}x")


if __name__ == "__main__":
    # Parameters for the benchmark
    VOCAB_SIZE = 50000
    EMBEDDING_DIM = 768
    BATCH_SIZES = [
        32,
    ]  # 64, 128, 256, 512, 1024, 2048, 4096]

    # Run the benchmark
    results = benchmark_embedding(VOCAB_SIZE, EMBEDDING_DIM, BATCH_SIZES)

    # Plot the results
    plot_results(results)

    # Test for correctness
    def test_correctness():
        print("\nTesting for correctness...")
        # Create small embeddings for testing
        torch_emb = nn.Embedding(100, 64).cuda()
        custom_emb = custom_embedding.CudaEmbedding(100, 64)

        # Copy weights to ensure same starting point
        with torch.no_grad():
            custom_emb.weight.copy_(torch_emb.weight)

        # Create test indices
        indices = torch.tensor([5, 10, 20, 30], dtype=torch.int32, device="cuda")

        # Forward pass

        torch_out = torch_emb(indices)
        custom_out = custom_emb(indices)

        # Check forward pass
        forward_diff = torch.abs(torch_out - custom_out).max().item()
        print(f"Max forward pass difference: {forward_diff:.8f}")

        # Backward pass
        target = torch.randn_like(torch_out)
        criterion = nn.MSELoss()

        # PyTorch backward
        torch_loss = criterion(torch_out, target)
        torch_loss.backward()
        torch_grad = torch_emb.weight.grad.clone()
        torch_emb.zero_grad()

        # Custom backward
        custom_loss = criterion(custom_out, target)
        custom_loss.backward()
        custom_grad = custom_emb.weight.grad.clone()
        custom_emb.zero_grad()

        # Check backward pass
        backward_diff = torch.abs(torch_grad - custom_grad).max().item()
        print(f"Max backward pass difference: {backward_diff:.8f}")

        if forward_diff < 1e-6 and backward_diff < 1e-6:
            print("✅ Correctness test passed!")
        else:
            print("❌ Correctness test failed!")

    # Run correctness test
    test_correctness()
