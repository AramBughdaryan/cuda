#include <iostream>
#include <cuda.h>
#include <torch/extension.h>
#include <cublas_v2.h>
#include <ATen/ATen.h>

class LinearFunction : public torch::autograd::Function<LinearFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input,
        const at::Tensor& weight,
        const at::Tensor& bias
    ) {
        TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
        TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
        
        int batch_size = input.size(0);
        int in_features = input.size(1);
        int out_features = weight.size(0);

        // Ensure input and weight match expected dimensions
        TORCH_CHECK(weight.size(1) == in_features, 
            "Weight dimensions do not match input features");

        // Create output tensor with same options as input
        auto output = torch::empty({batch_size, out_features}, input.options());

        // Create cublas handle with RAII-style management
        cublasHandle_t handle;
        TORCH_CHECK(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, 
            "Failed to create cuBLAS handle");
        
        // Unique pointer for automatic handle destruction
        std::unique_ptr<cublasContext, decltype(&cublasDestroy)> 
            handle_ptr(handle, cublasDestroy);

        float alpha = 1.0f;
        float beta = 0.0f;

        // Use AT_DISPATCH to handle different tensor types
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "LinearForward", ([&] {
            // Perform matrix multiplication
            cublasStatus_t status = cublasSgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                batch_size, out_features, in_features,
                &alpha,
                input.data_ptr<scalar_t>(), batch_size,
                weight.data_ptr<scalar_t>(), out_features,
                &beta,
                output.data_ptr<scalar_t>(), batch_size
            );
            
            TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, 
                "cublasSgemm failed in forward pass");
        }));

        if (bias.defined()) {
            TORCH_CHECK(bias.size(0) == out_features, 
                "Bias dimensions do not match output features");
            output.add_(bias);
        }

        ctx->save_for_backward({input, weight, bias});

        return output;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        // Retrieve saved tensors
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];

        // Extract gradient output
        auto grad_output = grad_outputs[0];

        int batch_size = input.size(0);
        int in_features = input.size(1);
        int out_features = weight.size(0);

        // Create gradient tensors
        auto grad_input = torch::empty({batch_size, in_features}, input.options());
        auto grad_weight = torch::empty({out_features, in_features}, weight.options());

        // Create cublas handle with RAII-style management
        cublasHandle_t handle;
        TORCH_CHECK(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, 
            "Failed to create cuBLAS handle");
        
        // Unique pointer for automatic handle destruction (smart pointer)
        std::unique_ptr<cublasContext, decltype(&cublasDestroy)> 
            handle_ptr(handle, cublasDestroy);

        float alpha = 1.0f;
        float beta = 0.0f;

        // Use AT_DISPATCH to handle different tensor types
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "LinearBackward", ([&] {
            // Compute gradient w.r.t. input
            cublasStatus_t status = cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                in_features, batch_size, out_features,
                &alpha, 
                weight.data_ptr<scalar_t>(), in_features,
                grad_output.data_ptr<scalar_t>(), out_features,
                &beta, 
                grad_input.data_ptr<scalar_t>(), in_features
            );
            TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, 
                "cublasSgemm failed for grad_input");

            // Compute gradient w.r.t. weight
            status = cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N,
                in_features, out_features, batch_size,
                &alpha, 
                input.data_ptr<scalar_t>(), batch_size,
                grad_output.data_ptr<scalar_t>(), batch_size,
                &beta, 
                grad_weight.data_ptr<scalar_t>(), in_features
            );
            TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, 
                "cublasSgemm failed for grad_weight");
        }));

        // Compute gradient w.r.t. bias
        auto grad_bias = bias.defined() ? grad_output.sum(0) : at::Tensor();

        return {grad_input, grad_weight, grad_bias};
    }
};

class LinearLayer : public torch::nn::Module {
private:
    at::Tensor weight;
    at::Tensor bias;
    int in_features;
    int out_features;

public:
    LinearLayer(int in_features, int out_features, bool with_bias = true)
    : in_features(in_features), out_features(out_features) {
        // Use torch::nn::init for more robust initialization
        weight = register_parameter("weight", 
            torch::empty({out_features, in_features}, 
                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32))
        );
        
        if (with_bias) {
            bias = register_parameter("bias", 
                torch::empty({out_features}, 
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32))
            );
        }
        
        reset_parameters();
    }

    void reset_parameters() {
        torch::nn::init::kaiming_uniform_(weight, std::sqrt(5.0));
        
        if (bias.defined()) {
            auto fan_in = weight.size(1);
            double bound = 1.0 / std::sqrt(fan_in);
            torch::nn::init::uniform_(bias, -bound, bound);
        }
    }

    at::Tensor forward(const at::Tensor& input) {
        // Ensure input is on CUDA
        TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
        
        return LinearFunction::apply(input, weight, bias);
    }
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_custom", 
        [](const at::Tensor& input, int in_features, int out_features) {
            LinearLayer layer(in_features, out_features);
            return layer.forward(input);
        }, 
        "Custom Linear Layer"
    );
}