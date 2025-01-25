#include <cuda_runtime.h>
#include <iostream>

__global__ void incrementKernel(int *data, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 2;
    }
}

int main(){
    const long int size = 5000000;
    int *hostArray = new int[size];
    int *deviceArray;

    for (int i = 0; i < size; i++) hostArray[i] = i;

    for (int i = 0; i < 30; i++) std::cout << hostArray[i] << '\t';

    std::cout << '\n';

    cudaMalloc((void **)&deviceArray, size * sizeof(int));

    cudaMemcpy(deviceArray, hostArray, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size * threadsPerBlock - 1) / threadsPerBlock;

    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceArray, size);

    cudaMemcpy(hostArray, deviceArray, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 30; i++) std::cout << hostArray[i] << '\t';
    
    cudaFree(deviceArray);
    delete [] hostArray;

    return 0;
}