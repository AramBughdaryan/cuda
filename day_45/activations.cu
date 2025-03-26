#include <iostream>
#include <cuda.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

// Global CUDA kernel for forward pass ReLU
__global__ void ReLUKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]); // Use fmaxf for better performance
    }
}

// Global CUDA kernel for backward pass ReLU
__global__ void ReLUBackwardKernel(
    const float* gradOutput, 
    const float* input, 
    float* gradInput, 
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Gradient is 1 if input > 0, 0 otherwise
        gradInput[idx] = (input[idx] > 0) ? gradOutput[idx] : 0.0f;
    }
}

class ReLU : public torch::autograd::Function<ReLU> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input
    ) {
        // Ensure input is contiguous and on CUDA
        TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
        
        int batch_size = input.size(0);
        int length = input.numel(); // Use numel() instead of size(1)
        
        // Create output tensor with same options as input
        auto output = torch::empty_like(input);
        
        // Compute grid and block dimensions
        const int blockSize = 256;
        const int numBlocks = (length + blockSize - 1) / blockSize;
        
        // Launch CUDA kernel

        ReLUKernel<<<numBlocks, blockSize>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            length
        );

        
        // Save input for backward pass
        ctx->save_for_backward({input});
        
        return output;
    }

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<torch::Tensor> grad_outputs
    ) {
        // Retrieve saved input
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        
        // Ensure grad_output is contiguous
        auto grad_output = grad_outputs[0].contiguous();
        
        // Create gradient input tensor
        auto grad_input = torch::empty_like(input);
        
        int length = input.numel(); // Use numel() 
        const int blockSize = 256;
        const int numBlocks = (length + blockSize - 1) / blockSize;
        
        // Launch CUDA kernel
        
        ReLUBackwardKernel<<<numBlocks, blockSize>>>(
            grad_output.data_ptr<float>(),
            input.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            length
        );
        
        return {grad_input};
    }
};

// Create a convenient function to register the custom autograd function to not export whole class
torch::Tensor relu_custom(const torch::Tensor& input) {
    return ReLU::apply(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_custom", &relu_custom, "Custom ReLU activation function");
}