#include "cuda_helpers.h"
#include <iostream>

template <typename T>
void printArray(const T* arr, int width, int height, const char* label) {
    std::cout << '\n' << label << ":\n";
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if constexpr (std::is_same_v<T, unsigned char>) {
                std::cout << static_cast<int>(arr[i * width + j]) << '\t';
            } else {
                std::cout << arr[i * width + j] << '\t';
            }
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);  // Optional: exit on error
    }
}

// Explicit template instantiations
template void printArray<unsigned char>(const unsigned char* arr, int width, int height, const char* label);
template void printArray<int>(const int* arr, int width, int height, const char* label);
template void printArray<double>(const double* arr, int width, int height, const char* label);
template void printArray<float>(const float* arr, int width, int height, const char* label);