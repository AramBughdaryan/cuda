#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void incrementKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;
    }
}

__host__ void increment(torch::Tensor tensor) {
    // Ensure the tensor is on the GPU and is contiguous
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    int size = tensor.numel(); // Total number of elements
    int *data = tensor.data_ptr<int>(); // Pointer to the data

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(data, size);

    // Wait for the kernel to complete
    cudaDeviceSynchronize();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("increment", &increment, "Increment array elements by 1 (CUDA)");
}


int main() {
    // Host is in our case CPU and device is GPU
    const int size = 1024;
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