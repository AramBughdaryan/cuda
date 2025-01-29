#include <iostream>
#include <cuda.h>

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
