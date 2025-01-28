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
        unsigned char g = Pin[rgbOffset + 2]; //TODO why we add +2 here instead of +1?!
        unsigned char b = Pin[rgbOffset + 3];

        Pout[greyOffset] = 0.21 * r + 0.71 * g + 0.07*b;
    }
}