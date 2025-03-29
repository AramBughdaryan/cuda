#pragma once
#include <simple_matrix_mul.h>

__global__ void simpleMatrixMulKernel(float* M, float* N, float* P, int width);