#include <cuda.h>
#include <iostream>

__shared__ float partialSum[SIZE];

partialSum[threadIdx.x] = X[blockIdx.x * blockDim.x + threadIdx.x];
unsigned int t = threadIdx.x;
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
{
    __syncthreads();
    if (t % (2 * stride) == 0)
        partialSum[t] += partialSum[t + stride];
}