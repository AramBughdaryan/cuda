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

        float* input_ptr = input.data_ptr<float>();
        float* weight_ptr = weight.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        ctx->save_for_backward({input, weight, bias});

        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }

        float alpha = 1.0f;
        float beta = 0.0f;

        status = cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            batch_size, out_features, in_features,
            &alpha, input_ptr, batch_size,
            weight_ptr, out_features, &beta,
            output_ptr, batch_size
        );
        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            throw std::runtime_error("cublasSgemm failed");
        }

        if (bias.defined()) {
            output.add_(bias);
        }

        // This part must be moved into cuda and either use cuBLAS or custom implementation
        cublasDestroy(handle);
        return output;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];

        auto grad_output = grad_outputs[0];

        int batch_size = input.size(0);
        int in_features = input.size(1);
        int out_features = weight.size(0);

        auto grad_input = torch::empty({batch_size, in_features}, input.options());
        auto grad_weight = torch::empty({out_features, in_features}, weight.options());

        float* grad_output_ptr = grad_output.data_ptr<float>();
        float* input_ptr = input.data_ptr<float>();
        float* weight_ptr = weight.data_ptr<float>();
        float* grad_input_ptr = grad_input.data_ptr<float>();
        float* grad_weight_ptr = grad_weight.data_ptr<float>();

        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }

        float alpha = 1.0f;
        float beta = 0.0f;

        // This is equivalent to: grad_input = torch::matmul(grad_output, weight);
        
        // grad_output is of shape (batch_size, out_features)
        // weight_ptr is of shape (out_features, in_features)
        status = cublasSgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            in_features, batch_size, out_features,
            &alpha, 
            weight_ptr, in_features,
            grad_output_ptr, out_features,
            &beta, 
            grad_input_ptr, in_features);

        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            throw std::runtime_error("cublasSgemm failed");
        }

        // grad_weight = grad_output.t() * input

        // gread_output is of shape [batch_size, out_features]
        // input is of shape [batch_size, in_features]
        status = cublasSgemm(handle, 
            CUBLAS_OP_T, CUBLAS_OP_N,
            in_features, out_features, batch_size,
            &alpha, 
            input_ptr, batch_size,
            grad_output_ptr, batch_size,
            &beta, 
            grad_weight_ptr, in_features);

        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            throw std::runtime_error("cublasSgemm failed");
        }

        auto grad_bias = grad_output.sum(0);

        cublasDestroy(handle);

        return {grad_input, grad_weight, grad_bias};
    }
};

class LinearLayer: public torch::nn::Module {
private:
    torch::Tensor weight;
    torch::Tensor bias;
    int in_features;
    int out_features;

public:
    LinearLayer(int in_features, int out_features, bool with_bias = true)
    : in_features(in_features), out_features(out_features) {
        weight = register_parameter("weight",
            torch::empty({out_features, in_features}, torch::kFloat32));
        
        if (with_bias){
            bias = register_parameter("bias",
            torch::empty({out_features}, torch::kFloat32));
        }
        reset_parameters();
    }

    void reset_parameters() {
        torch::nn::init::kaiming_uniform_(weight, std::sqrt(5.0));
        if (bias.defined()) {
            double bound = 1.0 / sqrt(in_features);
            torch::nn::init::uniform_(bias, -bound, bound);
        }
    }

    torch::Tensor forward(const torch::Tensor& input) {
        return LinearFunction::apply(input, weight, bias);
    }

};

int main() {
    // Your test code here.
    torch::Tensor input = torch::randn({2, 3}); // Example input
    LinearLayer layer(3, 4);
    torch::Tensor output = layer.forward(input);
    std::cout << output << std::endl;
    return 0;
}