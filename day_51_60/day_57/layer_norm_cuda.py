import torch
from torch.autograd import Function
import torch.nn as nn
import torch.cuda.nvrtc as nvrtc

# CUDA kernel for Layer Normalization with EMA
kernel_code = """
#include <cuda_runtime.h>
#include <math.h>

__global__ void layerNormEMAFwd(float *input, float *output, float *running_mean, float *running_std,
                                 int rows, int cols, float epsilon, float momentum,
                                 const float *gamma, const float *beta, float *mean_out, float *inv_std_out) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    __shared__ float shared_data[256];
    __shared__ float block_sum;
    __shared__ float block_sum_sq;

    int tid = threadIdx.x;
    int local_size = blockDim.x;

    // Load a segment of the row into shared memory
    int col_start = threadIdx.x;
    while (col_start < cols) {
        if (col_start < local_size) {
            shared_data[col_start] = input[row * cols + col_start];
        }
        col_start += blockDim.x;
    }
    __syncthreads();

    // Parallel reduction for sum
    float local_sum = (tid < cols) ? shared_data[tid] : 0.0f;
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < cols) {
            local_sum += shared_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sum = local_sum;
    }
    __syncthreads();
    float mean = block_sum / cols;

    // Parallel reduction for sum of squares
    float local_sum_sq = (tid < cols) ? (shared_data[tid] - mean) * (shared_data[tid] - mean) : 0.0f;
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < cols) {
            local_sum_sq += (shared_data[tid + s] - mean) * (shared_data[tid + s] - mean);
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sum_sq = local_sum_sq;
    }
    __syncthreads();
    float variance = block_sum_sq / cols;
    float inv_std = 1.0f / sqrtf(variance + epsilon);

    // Normalize and apply scale and bias
    col_start = threadIdx.x;
    while (col_start < cols) {
        if (col_start < local_size) {
            float normalized_value = (shared_data[col_start] - mean) * inv_std;
            output[row * cols + col_start] = normalized_value * (gamma ? gamma[col_start] : 1.0f) + (beta ? beta[col_start] : 0.0f);
            if (mean_out) mean_out[row * cols + col_start] = mean;
            if (inv_std_out) inv_std_out[row * cols + col_start] = inv_std;
        }
        col_start += blockDim.x;
    }

    // Update running mean and std (for the entire row) - only one thread per row does this
    if (tid == 0) {
        for (int c = 0; c < cols; ++c) {
            atomicExch(&running_mean[c], (1.0f - momentum) * running_mean[c] + momentum * mean);
            atomicExch(&running_std[c], (1.0f - momentum) * running_std[c] + momentum * sqrtf(variance + epsilon));
        }
    }
}

__global__ void layerNormEMABwd(float *grad_output, float *input, float *mean, float *inv_std,
                                 int rows, int cols,
                                 const float *gamma, float *grad_input, float *grad_gamma, float *grad_beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    __shared__ float shared_grad_output[256];
    __shared__ float shared_input[256];
    __shared__ float d_mean;
    __shared__ float d_inv_std;
    __shared__ float sum_dy;
    __shared__ float sum_dy_xmu;
    __shared__ float local_grad_gamma[32]; // Assuming max blockDim.x is multiple of 32
    __shared__ float local_grad_beta[32];

    int tid = threadIdx.x;
    int local_size = blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Load data into shared memory
    int col_start = threadIdx.x;
    while (col_start < cols) {
        if (col_start < local_size) {
            shared_grad_output[col_start] = grad_output[row * cols + col_start];
            shared_input[col_start] = input[row * cols + col_start];
        }
        col_start += blockDim.x;
    }
    __syncthreads();

    // Load mean and inv_std for the row
    if (tid == 0) {
        d_mean = mean[row * cols]; // Mean is the same across the row
        d_inv_std = inv_std[row * cols]; // Inv_std is the same across the row
    }
    __syncthreads();

    float dy = (tid < cols) ? shared_grad_output[tid] : 0.0f;
    float xmu = (tid < cols) ? (shared_input[tid] - d_mean) * d_inv_std : 0.0f;

    // Parallel reduction for sum_dy
    float local_sum_dy = dy;
    for (int offset = local_size / 2; offset > 0; offset >>= 1) {
        float neighbor = __shfl_xor_sync(0xffffffff, local_sum_dy, offset);
        if (tid < offset) {
            local_sum_dy += neighbor;
        }
    }
    if (tid % 32 == 0) {
        sum_dy = local_sum_dy;
    }

    // Parallel reduction for sum_dy_xmu
    float local_sum_dy_xmu = dy * xmu;
    for (int offset = local_size / 2; offset > 0; offset >>= 1) {
        float neighbor = __shfl_xor_sync(0xffffffff, local_sum_dy_xmu, offset);
        if (tid < offset) {
            local_sum_dy_xmu += neighbor;
        }
    }
    if (tid % 32 == 0) {
        sum_dy_xmu = local_sum_dy_xmu;
    }
    __syncthreads();

    float grad_norm = (sum_dy - xmu * sum_dy_xmu) * d_inv_std / cols;

    // Calculate grad_input
    if (tid < cols) {
        grad_input[row * cols + tid] = (dy - grad_norm - xmu * sum_dy_xmu * d_inv_std / cols) * d_inv_std * (gamma ? gamma[tid] : 1.0f);
    }

    // Calculate grad_gamma and grad_beta
    float local_dg = (tid < cols && gamma != nullptr) ? dy * (shared_input[tid] - d_mean) * d_inv_std : 0.0f;
    float local_db = (tid < cols) ? dy : 0.0f;

    // Warp-level reduction for grad_gamma and grad_beta
    for (int i = 16; i > 0; i >>= 1) {
        local_dg += __shfl_xor_sync(0xffffffff, local_dg, i);
        local_db += __shfl_xor_sync(0xffffffff, local_db, i);
    }

    if (lane_id == 0 && grad_gamma != nullptr && tid < cols) {
        atomicAdd(&grad_gamma[tid], local_dg);
    }
    if (lane_id == 0 && grad_beta != nullptr && tid < cols) {
        atomicAdd(&grad_beta[tid], local_db);
    }
}
"""

# Compile the CUDA kernel
nvrtc_program = nvrtc.Program(kernel_code, "layernorm_ema")
opts = ['-arch=sm_75']  # Adjust based on your GPU architecture
try:
    nvrtc_program.compile(opts)
except nvrtc.NVRTCError as error:
    print(error)
    exit()
module = torch.utils.dlpack.from_dlpack(nvrtc_program.get_ptx().encode().to_dlpack())

# Get the CUDA kernel functions
layernorm_ema_fwd_cuda = module.get_function("layerNormEMAFwd")
layernorm_ema_bwd_cuda = module.get_function("layerNormEMABwd")

class LayerNormEMAFunction(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_std, weight, bias, epsilon, momentum):
        rows, cols = input.shape
        output = torch.empty_like(input)
        mean_out = torch.empty_like(input)
        inv_std_out = torch.empty_like(input)

        threads_per_block = 256
        blocks_per_row = (cols + threads_per_block - 1) // threads_per_block
        grid = (blocks_per_row, rows)
        block = (threads_per_block, 1)

        layernorm_ema_fwd_cuda(
            grid=grid,
            block=block,
            args=[
                input.data_ptr(),
                output.data_ptr(),
                running_mean.data_ptr(),
                running_std.data_ptr(),
                torch.tensor([rows], dtype=torch.int32).cuda(),
                torch.tensor([cols], dtype=torch.int32).cuda(),
                torch.tensor([epsilon], dtype=torch.float32).cuda(),
                torch.tensor([momentum], dtype=torch.float32).cuda(),
                weight.data_ptr() if weight is not None else None,
                bias.data_ptr() if bias is not None else None,
                mean_out.data_ptr(),
                inv_std_out.data_ptr()
            ],
            stream=torch.cuda.current_stream()
        )

        ctx.save_for_backward(input, weight, mean_out, inv_std_out)
        ctx.epsilon = epsilon
        ctx.momentum = momentum
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, mean, inv_std = ctx.saved_tensors
        rows, cols = input.shape
        grad_input = torch.empty_like(input)
        grad_weight = torch.zeros_like(weight) if weight is not None else None
        grad_bias = torch.zeros_like(weight) if weight is not None else None

        threads_per_block = 256
        blocks_per_row = (cols + threads_per_block - 1) // threads_per_block
        grid = (blocks_per_row, rows)
        block = (threads_per_block, 1)

        layernorm_ema_bwd_cuda(
            grid=grid,
            block=block,
            args=[
                grad_output.data_ptr(),
                input.data_ptr(),
                mean.data_ptr(),
                inv_std.data_ptr(),
                torch.tensor([rows], dtype=torch.int32).cuda(),
                torch.tensor([cols], dtype=torch.int32).cuda(),
                weight.data_ptr() if weight is not None else None,
                grad_input.data_ptr(),
                grad_weight.data_ptr() if grad_weight is not None else None,
                grad_bias.data_ptr() if grad_bias is not None else None
            ],
            stream=torch.cuda.current_stream()
        )

        return grad_input, None, None, grad_weight, grad_bias, None, None
