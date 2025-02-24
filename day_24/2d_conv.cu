#include <cuda.h>
#include <iostream>

#include "../helpers/cuda_helpers.h"

#define TILE_WIDTH 5
#define MAX_MASK_WIDTH 9
#define O_TILE_WIDTH (TILE_WIDTH - MAX_MASK_WIDTH + 1)

// Fix the typo in typedef and structure definition
typedef struct {
    int width;
    int height;
    int pitch;
    int channels;
    float* data;
} ImageStruct, *wbImage_t;

// Define mask data
__constant__ float M[MAX_MASK_WIDTH * MAX_MASK_WIDTH];

__global__ void convolution_2D_tiled_kernel(float *P, float *N, int height, int width,
                                          int pitch, int channels, int Mask_Width,
                                          const float* __restrict__ M)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * O_TILE_WIDTH + ty;
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;

    int row_i = row_o - Mask_Width/2;
    int col_i = col_o - Mask_Width/2;

    __shared__ float N_ds[TILE_WIDTH + MAX_MASK_WIDTH - 1][TILE_WIDTH + MAX_MASK_WIDTH - 1];

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)){
        N_ds[ty][tx] = N[row_i * pitch + col_i];
    }
    else{
        N_ds[ty][tx] = 0.0f;
    }
    
    __syncthreads();

    float output = 0.0f;
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH){
        for (int i = 0; i < Mask_Width; i++){
            for (int j = 0; j < Mask_Width; j++){
                output += M[i * Mask_Width + j] * N_ds[i + ty][j + tx];
            }
        }
        if (row_o < height && col_o < width){
            P[row_o * width + col_o] = output;
        }
    }
}

int main() {
    // Example dimensions
    int width = 1024;
    int height = 1024;
    int pitch = width;  // For simplicity
    int channels = 1;   // Grayscale
    int mask_width = 3; // 3x3 mask

    // Allocate host memory
    float *h_output = new float[height * width];
    float *h_input = (float*)malloc(height * width * sizeof(float));
    float *h_mask = (float*)malloc(mask_width * mask_width * sizeof(float));

    initializeMatrix(h_mask, mask_width, mask_width);


    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, height * width * sizeof(float));
    cudaMalloc(&d_output, height * width * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, h_mask, mask_width * mask_width * sizeof(float));

    // Calculate grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + O_TILE_WIDTH - 1) / O_TILE_WIDTH, 
                 (height + O_TILE_WIDTH - 1) / O_TILE_WIDTH);

    // Launch kernel
    convolution_2D_tiled_kernel<<<dimGrid, dimBlock>>>(d_output, d_input, height, width, 
                                                      pitch, channels, mask_width, M);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    printArray(h_mask, mask_width, mask_width, "mask_width");
    printArray(h_output, width, height, "h_output");
    delete[] h_input;
    delete[] h_output;
    delete[] h_mask;

    return 0;
}