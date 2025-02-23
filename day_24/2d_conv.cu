#include <cuda.h>
#include <iostream>

#include "../helpers/cuda_helpers.h"

typdef struct {
    int width;
    int height;
    int pitch;
    int channels;
    float* data;
} *  wbImage_t;

__global__ void convolution_2D_tiled_kernel(float *P, float *N, int height, int width,
                                            int pitch, int channels, int Mask_Width,
                                            const float __restrict__ *M)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * O_TILE_WIDTH + ty;
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;

    int row_i = row_o - Mask_Width/2;
    int col_i = col_o - Mask_Width/2;

}