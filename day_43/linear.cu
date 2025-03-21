#include <iostream>
#include <cuda.h>
#include <torch/extension.h>

class LinearFunction : public torch::autograd::Function<LinearFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input,
        const at::Tensor& weight,
        const at::Tensor& bias
    ) {
        ctx->save_for_backward({input, weight, bias});
        // This part must be moved into cuda and either use cuBLAS or custom implementation
        auto output = torch::matmul(input, weight.t());
        if (bias.defined()) {
            output.add_(bias);
        }
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