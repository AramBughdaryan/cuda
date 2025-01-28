#include <iostream>
#include <cuda.h>

__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

__host__ void vecAdd(float* h_A, float* h_B, float* h_C, int n)
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
    // Here ceil(size / 256.0) is number of blocks
    // 256 as second argument is the number of threads in each block
    vecAddKernel<<<ceil(size / 256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main(){
    const int n = 1024;
    int i;
    float *h_A = new float[n];
    float *h_B = new float[n];
    float *h_C = new float[n];

    for (i = 0; i < n; i++){
        h_A[i] = i + 4;
        h_B[i] = pow(i, 3);
    }
    vecAdd(h_A, h_B, h_C, n);
    for (i = 0; i < 10; i++) std::cout<<h_C[i]<<' ';

}