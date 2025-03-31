Here's an improved README with better formatting, explanations, and additional details about cuBLAS usage:

# CUDA Linear Layer Implementation

This project implements a custom linear (fully connected) layer using CUDA and cuBLAS for accelerated matrix operations. The implementation includes both forward and backward passes for autograd compatibility with PyTorch.

## Features

- Custom CUDA implementation of a linear layer
- Integration with PyTorch's autograd system
- cuBLAS-accelerated matrix multiplication
- Support for optional bias terms
- Proper weight initialization (Kaiming uniform for weights, uniform for bias)

## Implementation Details

### LinearFunction Class

The `LinearFunction` class implements the forward and backward passes using cuBLAS:

1. **Forward Pass**:
   - Uses `cublasSgemm` for matrix multiplication (input × weight^T)
   - Adds bias if defined
   - Saves input, weight, and bias for backward pass

2. **Backward Pass**:
   - Computes gradients for input, weight, and bias
   - Uses `cublasSgemm` for all matrix operations
   - Properly handles transpositions for gradient calculations

### LinearLayer Class

The `LinearLayer` class provides a PyTorch Module interface:

- Manages weight and bias parameters
- Implements proper initialization (Kaiming uniform for weights)
- Provides `reset_parameters()` method
- Wraps the `LinearFunction` for forward operations

## cuBLAS Usage

This implementation leverages cuBLAS (`cublasSgemm`) for all matrix operations. Key aspects:

1. **Matrix Multiplication**:
   - Forward pass: `output = input × weight^T`
   - Input gradient: `grad_input = grad_output × weight`
   - Weight gradient: `grad_weight = grad_output^T × input`

2. **Operation Parameters**:
   - `CUBLAS_OP_N`: No transposition
   - `CUBLAS_OP_T`: Transposition
   - Alpha and beta parameters control scaling of input/output matrices

### Example cuBLAS SGEMM Call

```cpp
float alpha = 1.0f;
float beta = 0.0f;
cublasSgemm(
    handle,          // cuBLAS handle
    CUBLAS_OP_N,     // Don't transpose A
    CUBLAS_OP_T,     // Transpose B
    batch_size,      // Rows of A
    out_features,    // Columns of B^T
    in_features,     // Columns of A / Rows of B
    &alpha,          // Scaling factor
    input_ptr,       // Matrix A
    batch_size,      // Leading dimension of A
    weight_ptr,      // Matrix B
    out_features,    // Leading dimension of B
    &beta,           // Scaling factor for C
    output_ptr,      // Matrix C
    batch_size       // Leading dimension of C
);
```

## Compilation

To compile this code, you'll need to link against several libraries:

```bash
nvcc linear.cu -o linear \
-I/path/to/torch/include \
-I/path/to/torch/include/torch/csrc/api/include \
-I/path/to/python/include \
-L/path/to/torch/lib \
-lc10 -ltorch -ltorch_cpu -lcublas -lcudart -lc10_cuda -ltorch_cuda
```

You may also need to set the library path:

```bash
export LD_LIBRARY_PATH=/path/to/torch/lib:$LD_LIBRARY_PATH
```

## Example Usage

```cpp
int main() {
    // Create random input tensor (batch_size=128, in_features=32)
    at::Tensor input = torch::randn({128, 32});
    
    // Create linear layer (in_features=32, out_features=4)
    LinearLayer layer(32, 4);
    
    // Forward pass
    at::Tensor output = layer.forward(input);
    
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Sample output:\n" << output.slice(0, 0, 5) << std::endl;
    
    return 0;
}
```

## Performance Notes

- The cuBLAS implementation provides highly optimized matrix operations
- Memory layout is important for performance (row-major vs column-major)
- The current implementation uses single-precision floating-point (float32)
- For best performance, ensure input tensors are contiguous in memory

## Dependencies

- CUDA Toolkit
- cuBLAS
- PyTorch with CUDA support
- C++17 compatible compiler