#include <iostream>
#include <cuda.h>

__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x
    if (i < n) C[i] = A[i] + B[i];
}


void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaError_t err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);



}

int main(){
    const int n = 1024;
    float *h_A;
    float *h_B;
    float *h_C;

    float *d_A;
    float *d_B;
    float *d_C;
    int size = n * sizeof(float);

    

    

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}