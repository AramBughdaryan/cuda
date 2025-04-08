#include <cuda_runtime.h>
#include <stdio.h>

__global__ void layerNormKernel(float *input, float *output, int rows, int cols, float epsilon, float *gamma, float *beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    __shared__ float shared_data[256]; // Shared memory for a block's row
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
    float stddev = sqrtf(variance + epsilon);

    // Normalize and apply scale and bias
    col_start = threadIdx.x;
    while (col_start < cols) {
        if (col_start < local_size) {
            float normalized_value = (shared_data[col_start] - mean) / stddev;
            output[row * cols + col_start] = normalized_value * (gamma ? gamma[col_start] : 1.0f) + (beta ? beta[col_start] : 0.0f);
        }
        col_start += blockDim.x;
    }
}

int main() {
    int rows = 2;
    int cols = 4;
    float epsilon = 1e-5f;

    // Allocate host memory
    float *h_input = (float *)malloc(rows * cols * sizeof(float));
    float *h_output = (float *)malloc(rows * cols * sizeof(float));
    float *h_gamma = (float *)malloc(cols * sizeof(float));
    float *h_beta = (float *)malloc(cols * sizeof(float));

    // Initialize host data
    h_input[0] = 1.0f; h_input[1] = 2.0f; h_input[2] = 3.0f; h_input[3] = 4.0f;
    h_input[4] = 5.0f; h_input[5] = 6.0f; h_input[6] = 7.0f; h_input[7] = 8.0f;

    h_gamma[0] = 0.5f; h_gamma[1] = 1.0f; h_gamma[2] = 1.5f; h_gamma[3] = 2.0f;
    h_beta[0] = 0.1f; h_beta[1] = 0.2f; h_beta[2] = 0.3f; h_beta[3] = 0.4f;

    // Allocate device memory
    float *d_input, *d_output, *d_gamma, *d_beta;
    cudaMalloc((void **)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void **)&d_output, rows * cols * sizeof(float));
    cudaMalloc((void **)&d_gamma, cols * sizeof(float));
    cudaMalloc((void **)&d_beta, cols * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, cols * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerRow = (cols + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(blocksPerRow, rows);

    // Launch the kernel
    layerNormKernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols, epsilon, d_gamma, d_beta);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output
    printf("Input:\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", h_input[i * cols + j]);
        }
        printf("\n");
    }

    printf("Output (Layer Normalized):\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", h_output[i * cols + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    // Free host memory
    free(h_input);
    free(h_output);
    free(h_gamma);
    free(h_beta);

    return 0;
}