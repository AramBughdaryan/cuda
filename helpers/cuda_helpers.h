// cuda_helpers.h
#pragma once
#include <cuda_runtime.h>

// Template declaration for printing arrays
template <typename T>
void printArray(const T* arr, int width, int height, const char* label);

// Check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg);

template <typename T>
void initializeMatrices(T* A, T* B, int M, int K, int N);

template <typename T>
void initializeMatrix(T* A, int M, int K);