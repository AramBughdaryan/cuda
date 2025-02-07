#include "cuda_helpers.h"
#include <iostream>
#include <iomanip>

template <typename T>
void printArray(const T* arr, int width, int height, const char* label) {
    std::cout << '\n' << label << ":\n";
    std::cout << std::fixed;
    std::cout << std::setprecision(6);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if constexpr (std::is_same_v<T, unsigned char>) {
                std::cout << std::setw(12) << static_cast<int>(arr[i * width + j]) << '\t';
            } else {
                std::cout << std::setw(12) << arr[i * width + j] << '\t';
            }
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}
// Explicit template instantiations
template void printArray<unsigned char>(const unsigned char* arr, int width, int height, const char* label);
template void printArray<int>(const int* arr, int width, int height, const char* label);
template void printArray<double>(const double* arr, int width, int height, const char* label);
template void printArray<float>(const float* arr, int width, int height, const char* label);


void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);  // Optional: exit on error
    }
}

template <typename T>
void initializeMatrices(T* A, T* B, int M, int K, int N) {
    for(int i = 0; i < M * K; i++) A[i] = rand() / (T)RAND_MAX;
    for(int i = 0; i < K * N; i++) B[i] = rand() / (T)RAND_MAX;
}
template void initializeMatrices<float>(float* A, float* B, int M, int K, int N);
template void initializeMatrices<int>(int* A, int* B, int M, int K, int N);
template void initializeMatrices<double>(double* A, double* B, int M, int K, int N);

template <typename T>
void initializeMatrix(T* A, int M, int K) {
    for(int i = 0; i < M * K; i++) A[i] = rand() / (T)RAND_MAX;
}
template void initializeMatrix<float>(float* A, int M, int K);
template void initializeMatrix<int>(int* A, int M, int K);
template void initializeMatrix<double>(double* A, int M, int K);