#include <cuda.h>
#include <iostream>

__global__ void histo_kernel(char *data, usinged int length, unsinged int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i , length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo[alphabet_position/4], 1);
        }
    }
}