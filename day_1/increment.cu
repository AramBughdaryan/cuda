#include <cuda_runtime.h>
#include <iostream>

// Kernel function to increment array elements
__global__ void incrementKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;
    }
}

int main() {
    // Host is in our case CPU and device is GPU
    const int size = 2048;
    int *hostArray = new int[size];
    int *deviceArray;

    // Initialize host array
    for (int i = 0; i < size; i++) hostArray[i] = i;

    // Allocate memory on the GPU
    cudaMalloc((void **)&deviceArray, size * sizeof(int));

    // Copy data to the GPU
    cudaMemcpy(deviceArray, hostArray, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceArray, size);

    // Copy result back to the host
    cudaMemcpy(hostArray, deviceArray, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < 10; i++) std::cout << hostArray[i] << " ";
    std::cout << std::endl;

    // Clean up
    cudaFree(deviceArray);
    delete[] hostArray;

    return 0;
}