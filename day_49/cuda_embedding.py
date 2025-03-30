import torch
import torch.nn as nn
from torch.autograd import Function
import time
import os

from torch.utils.cpp_extension import load, include_paths as torch_include_paths

nvcc_compile_args = [
            "-O3", "--ftz=true", "-Xptxas=-O3", "--maxrregcount=64", "--default-stream=per-thread",
            "--restrict", "-lineinfo",
            *[f"-I{path}" for path in torch_include_paths()],
        ]
# Import the C++ extension
# Note: This assumes the above C++ code is compiled as a module named "cuda_embedding"

    
    
    # Just-in-time compile the extension
    # You can skip this if you've already built the extension with setup.py
cuda_embedding = load(
    name="cuda_embedding",
    sources=["embedding_layer.cu"],
    verbose=True,
    extra_cuda_cflags=nvcc_compile_args,
    with_cuda=True,
)


def benchmark_embedding(vocab_size, embedding_dim, batch_size, num_runs=100):
    """
    Benchmark custom CUDA embedding against PyTorch's embedding
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of each embedding
        batch_size: Number of indices to look up
        num_runs: Number of iterations for timing
    
    Returns:
        dict: Dictionary with benchmark results
    """
    print(f"\nBenchmarking with vocab_size={vocab_size}, embedding_dim={embedding_dim}, batch_size={batch_size}")
    
    # Create input data
    indices = torch.randint(0, vocab_size, (batch_size,), device="cuda")
    
    # Create embedding layers
    custom_embedding = cuda_embedding.CudaEmbedding(vocab_size, embedding_dim)
    pytorch_embedding = nn.Embedding(vocab_size, embedding_dim).cuda()
    custom_embedding.set_weight(pytorch_embedding.weight.data) 
    import pdb; pdb.set_trace()
    # Ensure identical weights for fair comparison
    
    grad_output = torch.randn(batch_size, embedding_dim, device="cuda")
    
    # Warmup
    for _ in range(10):
        custom_out = custom_embedding.forward(indices.to(torch.int32))
        pytorch_out = pytorch_embedding(indices)
    torch.cuda.synchronize()
    
    # Benchmark forward pass - Custom
    start_time = time.time()
    for _ in range(num_runs):
        custom_out = custom_embedding.forward(indices.to(torch.int32))
        torch.cuda.synchronize()
    custom_forward_time = (time.time() - start_time) * 1000 / num_runs  # ms
    
    # Benchmark forward pass - PyTorch
    start_time = time.time()
    for _ in range(num_runs):
        pytorch_out = pytorch_embedding(indices)
        torch.cuda.synchronize()
    pytorch_forward_time = (time.time() - start_time) * 1000 / num_runs  # ms
    
    # Prepare for backward pass
    custom_embedding.zero_grad()
    pytorch_embedding.zero_grad()
    
    # Benchmark backward pass - Custom
    start_time = time.time()
    for _ in range(num_runs):
        out = custom_embedding.forward(indices.to(torch.int32))
        loss = (out * grad_output).sum()
        loss.backward()
        torch.cuda.synchronize()
    custom_backward_time = (time.time() - start_time) * 1000 / num_runs  # ms
    
    # Save gradients for comparison
    custom_grads = custom_embedding.weight.grad.clone()
    custom_embedding.zero_grad()
    
    # Benchmark backward pass - PyTorch
    start_time = time.time()
    for _ in range(num_runs):
        out = pytorch_embedding(indices)
        loss = (out * grad_output).sum()
        loss.backward()
        torch.cuda.synchronize()
    pytorch_backward_time = (time.time() - start_time) * 1000 / num_runs  # ms
    
    # Save gradients for comparison
    pytorch_grads = pytorch_embedding.weight.grad.clone()
    
    # Check correctness
    forward_diff = (custom_out - pytorch_out).abs().max().item()
    backward_diff = (custom_grads - pytorch_grads).abs().max().item()
    
    # Print results
    print("\nResults:")
    print(f"Forward pass:")
    print(f"  Custom implementation: {custom_forward_time:.4f} ms")
    print(f"  PyTorch implementation: {pytorch_forward_time:.4f} ms")
    print(f"  Speedup: {pytorch_forward_time / custom_forward_time:.2f}x")
    
    print(f"\nBackward pass:")
    print(f"  Custom implementation: {custom_backward_time:.4f} ms")
    print(f"  PyTorch implementation: {pytorch_backward_time:.4f} ms")
    print(f"  Speedup: {pytorch_backward_time / custom_backward_time:.2f}x")
    
    print(f"\nCorrectness check:")
    print(f"  Forward difference: {forward_diff:.8f}")
    print(f"  Backward difference: {backward_diff:.8f}")
    print(f"  Forward match: {'YES' if forward_diff < 1e-5 else 'NO'}")
    print(f"  Backward match: {'YES' if backward_diff < 1e-5 else 'NO'}")
    
    return {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "custom_forward_time": custom_forward_time,
        "pytorch_forward_time": pytorch_forward_time,
        "custom_backward_time": custom_backward_time,
        "pytorch_backward_time": pytorch_backward_time,
        "forward_speedup": pytorch_forward_time / custom_forward_time,
        "backward_speedup": pytorch_backward_time / custom_backward_time,
        "forward_diff": forward_diff,
        "backward_diff": backward_diff
    }


if __name__ == "__main__":
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        exit(1)
    
    print("CUDA Embedding Benchmark vs PyTorch Embedding")
    
    # Run benchmarks with different configurations
    configs = [
        {"vocab_size": 1000, "embedding_dim": 64, "batch_size": 128},
        {"vocab_size": 10000, "embedding_dim": 256, "batch_size": 512},
        {"vocab_size": 50000, "embedding_dim": 512, "batch_size": 1024}
    ]
    
    results = []
    for config in configs:
        result = benchmark_embedding(**config)
        results.append(result)
    
    # Print summary
    print("\n=== Summary ===")
    print("Configuration | Forward Speedup | Backward Speedup | Forward Match | Backward Match")
    for r in results:
        print(f"V:{r['vocab_size']},D:{r['embedding_dim']},B:{r['batch_size']} | "
              f"{r['forward_speedup']:.2f}x | {r['backward_speedup']:.2f}x | "
              f"{'✓' if r['forward_diff'] < 1e-5 else '✗'} | "
              f"{'✓' if r['backward_diff'] < 1e-5 else '✗'}")