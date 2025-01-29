#include "cuda_helpers.h"
#include <iostream>


void printArray(const unsigned char* arr, int widht, int height, const char* label){
    std::cout<<'\n'<<label<<":\n";
    for (int i = 0; i < height; i++){
        for (int j = 0; j < widht; j++){
            std::cout<<static_cast<int>(arr[i * widht + j])<<'\t';
        }
        std::cout<<'\n';
    }
    std::cout<<'\n';
}

void checkCudaError(cudaError_t err, const char* msg){
    if (err != cudaSuccess){
        std::cerr<<"CUDA Error at "<<msg<<": "<<cudaGetErrorString(err)<<std::endl;
    }
}
