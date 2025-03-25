#include <iostream>
#include <cuda.h>
#include <torch/extension.h>
#include <cublas_v2.h>
#include <ATen/ATen.h>

__global__ void ReLUKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(0.0f, input[idx]);
    }
}

__global__ void ReLUBackwardKernel(
    float* gradOutput, 
    float* input, 
    float* gradInput, 
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Gradient is 1 if input > 0, 0 otherwise
        gradInput[idx] = input[idx] > 0 ? gradOutput[idx] : 0.0f;
    }
}

__global__ void leakyReluKernel(float* input, float* output, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : alpha * input[idx];
    }
}

__global__ void leakyReluBackwardKernel(
    float* gradOutput, 
    float* input, 
    float* gradInput, 
    int size, 
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Gradient is 1 if input > 0, alpha otherwise
        gradInput[idx] = input[idx] > 0 ? gradOutput[idx] : alpha * gradOutput[idx];
    }
}

__global__ void sigmoidKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void sigmoidBackwardKernel(
    float* gradOutput, 
    float* output, 
    float* gradInput, 
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Derivative of sigmoid: output * (1 - output)
        gradInput[idx] = gradOutput[idx] * output[idx] * (1.0f - output[idx]);
    }
}


class ReLU : public torch::autograd::Function<ReLU> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input,
    ) {
    int batch_size = input.size(0);
    int length = input.size(1);
    auto output = torch::empty({batch_size, out_features}, input.options());

    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    ctx->save_for_backward({input});
    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;
    ReLUKernel<<<numBlocks, blockSize>>>(input, output, length);

    return output;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];

        int blockSize = 256;
        int numBlocks = (length + blockSize - 1) / blockSize;

        // declare gradInput and pass to backward kernel

        ReLUBackwardKernel<<<numBlocks, blockSize>>>(grad_outputs, input);

        

        
    }

};