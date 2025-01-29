#pragma once
#include <cuda_runtime.h>

// Print a 2D array with a label
void printArray(const unsigned char* arr, int widht, int height, const char* label);

// Check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg);