#include <cuda.h>
#include <iostream>

__global__ void histo_kernel(char *data, usinged int length, unsinged int *histo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int section_size = (length - 1) / (blockDim.x * gridDim.x) + 1;
    int start = i * section_size;
    
    for (int k =0; k < section_size; k++) {
        if (start + k < length) {
            int alphabet_position = data[start + k] - 'a';
            if (alphabet_position >= 0 && alphabet_position < 26) {
                atomicAdd(&histo[alphabet_position/4], 1);
            }
        }
    }
}