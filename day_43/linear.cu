#include <iostream>
#include <cuda.h>
#include <torch/extension.h>
#include <cublas_v2.h>


class LinearFunction : public torch::autograd::Function<LinearFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input,
        const at::Tensor& weight,
        const at::Tensor& bias
    ) {
        int batch_size = input.size(0);
        int in_features = input.size(1);
        int out_features = weight.size(0);

        auto output = torch::empty({batch_size, out_features}, input.options());

        const float* input_ptr = input.data_ptr<float>();
        const float* weight_ptr = weight.data_ptr<float>();
        const float* output_ptr = output.data_ptr<float>();

        ctx->save_for_backward({input, weight, bias});

        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;

        cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, batch_size,
            out_features, in_features, &alpha, input_ptr, batch_size, weight_ptr, out_features,
            &beta, output_ptr, batch_size
        )

        if (bias.defined()) {
            output.add_(bias);
        }

        // This part must be moved into cuda and either use cuBLAS or custom implementation
        auto output = torch::matmul(input, weight.t());
        if (bias.defined()) {
            output.add_(bias);
        }
        cudaDestroy(handle);
        return output;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_tensors();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];

        auto grad_output = grad_outputs[0];

        auto grad_input = torch::matmul(grad_output, weight);
        auto grad_weight = torch::matmul(grad_output.t(), input);
        auto grad_bias = grad_output.sum(0);

        return {grad_input, grad_weight, grad_bias};
    }
};