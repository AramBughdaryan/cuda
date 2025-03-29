#include <iostream>
#include <cuda.h>

constexpr int CHANNELS = 3;


__global__ void colorToGreyscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height){
    int Col = threadIdx.x + blockDim.x * blockIdx.x;
    int Row = threadIdx.y + blockDim.y * blockIdx.y;

    if (Col < width && Row < height){
        int greyOffset = Row * width + Col;
        int rgbOffset = greyOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset]; // red value pixel
        unsigned char g = Pin[rgbOffset + 1]; // green value pixel
        unsigned char b = Pin[rgbOffset + 2]; // blue value pixel

        Pout[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}


void testKernel(){
    int height = 4, width = 4;
    int imgSize = width * height * CHANNELS * sizeof(unsigned char);
    int greySize = width * height * sizeof(unsigned char);

    unsigned char h_Pin[] = {
        255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255,
        100, 50, 25, 200, 150, 100, 50, 25, 10,  75, 75, 75,
        30, 60, 90,  180, 200, 220, 120, 130, 140,  10, 20, 30,
        240, 128, 64,  32, 16, 8,  255, 255, 0,  128, 128, 128
    };

    unsigned char h_Pout[width * height];
    unsigned char *d_Pin, *d_Pout;

    cudaMalloc((void**)&d_Pin, imgSize);
    cudaMalloc((void**)&d_Pout, greySize);

    cudaMemcpy(d_Pin, h_Pin, imgSize, cudaMemcpyHostToDevice);

    dim3 blockSize(2, 2);
    dim3 gridSize(ceil(width / blockSize.x),
                  ceil(height / blockSize.y));

    std::cout<<ceil(width / blockSize.x);

    colorToGreyscaleConversion<<<gridSize, blockSize>>>(d_Pout, d_Pin, width, height);

    cudaMemcpy(h_Pout, d_Pout, greySize, cudaMemcpyDeviceToHost);

    std::cout << "Grayscale Output:\n";
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << (int)h_Pout[i * width + j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_Pin);
    cudaFree(d_Pout);
}

int main(){
    testKernel();
}