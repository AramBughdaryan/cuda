#include <iostream>

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    for (int i = 0; i < n; i++) h_C[i] = h_A[i] + h_B[i];
}

int main()
{
    const int n = 5000;
    float *h_A = new float[n];
    float *h_B = new float[n];
    float *h_C = new float[n];

    for (int i = 0; i < n ; i ++) {
        h_B[i] = i;
        h_C[i] = i ^ 2;
    }

    vecAdd(h_A, h_B, h_C, n);
    for (int i = 0; i < n; i++) std::cout<<h_C[i]<<' ';

}