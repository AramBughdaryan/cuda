#include <iostream>
#include <cuda.h>
#include <vector>
#include <cassert>
#include "../helpers/cuda_helpers.h"

constexpr int BLUR_SIZE = 1; // for 3 by 3 box

__global__ void blurKernel(unsigned char* in, unsigned char* out, int w, int h){
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h){
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
        {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++ blurCol)
            {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
                {   
                    pixVal += in[curRow * w + curCol];
                    pixels ++;
                }
            }
        }
        out[Row * w + Col] = (unsigned char)(pixVal / pixels);
    }
}

__host__ void testBlurKernel(){
    const int width = 4;
    const int height = 4;
    const int size = width * height * sizeof(unsigned char);

      unsigned char h_input[width * height] = {
        10, 10, 10, 10,
        10, 255, 255, 10,
        10, 255, 255, 10,
        10, 10, 10, 10
    };

    unsigned char h_output[width * height] = {0};

    unsigned char *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, size), "Input allocation");
    checkCudaError(cudaMalloc(&d_output, size), "Output allocation");

    checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice), "Copy input array from Host to Device");

    dim3 blockSize(2, 2);
    dim3 gridSize(ceil(width/blockSize.x), ceil(height/blockSize.y));

    blurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    
    checkCudaError(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost), "Copy output from Device to Host");

    printArray(h_input, width, height, "Input Array");
    printArray(h_output, width, height, "Blurred Output");
    
    bool hasChange = false;
    for (int i = 0; i < width * height; i++) {
        if (h_input[i] != h_output[i]) {
            hasChange = true;
            break;
        }
    }
    std::cout << "Blur test " << (hasChange ? "PASSED" : "FAILED") 
              << " - Values were " << (hasChange ? "" : "not ") << "modified\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    testBlurKernel();
    return 0;
}

// Make sure to include cuda_helpers when compileing. Use
// nvcc -I../helpers image_blur.cu ../helpers/cuda_helpers.cpp -o blur_test