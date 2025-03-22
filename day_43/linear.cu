#include <iostream>
#include <cuda.h>
#include <torch/extension.h>

__global__ void tiledMatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Saved as automatic variables thus in registers. 
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // As we declare this variable as automatic it will be private for each thread!
    float Pvalue = 0;
    for (int phase = 0; phase < Width / TILE_WIDTH; ++phase){
        // Collaborative loading of d_M and d_N tiles into shared memory
        Mds[ty][tx] = d_M[Row * Width + phase * TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(phase * TILE_WIDTH + threadIdx.y) * Width + Col]; 
        __syncthreads();

        for (int k =0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    d_P[Row * Width + Col] = Pvalue;
}



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