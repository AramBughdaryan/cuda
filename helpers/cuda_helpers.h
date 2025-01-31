// cuda_helpers.h
#pragma once
#include <cuda_runtime.h>

// Template declaration for printing arrays
template <typename T>
void printArray(const T* arr, int width, int height, const char* label);

// Check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg);
