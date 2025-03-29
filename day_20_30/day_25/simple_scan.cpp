#include <iostream>
#include <cassert>

void sequential_scan(float *x, float *y, int Max_i) {
    int accumulator = x[0];
    y[0] = accumulator;
    for (int i = 1; i < Max_i; i++) {
        accumulator += x[i];
        y[i] = accumulator;
    }
}



void test_sequential_scan() {
    // Test 1: Basic test with positive numbers
    float x1[] = {1.0, 2.0, 3.0, 4.0};
    float y1[4] = {0};
    sequential_scan(x1, y1, 4);
    assert(y1[0] == 1.0);
    assert(y1[1] == 3.0);
    assert(y1[2] == 6.0);
    assert(y1[3] == 10.0);

    // Test 2: Test with negative numbers
    float x2[] = {-1.0, -2.0, -3.0};
    float y2[3] = {0};
    sequential_scan(x2, y2, 3);
    assert(y2[0] == -1.0);
    assert(y2[1] == -3.0);
    assert(y2[2] == -6.0);

    // Test 3: Single element array
    float x3[] = {5.0};
    float y3[1] = {0};
    sequential_scan(x3, y3, 1);
    assert(y3[0] == 5.0);

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    test_sequential_scan();
    return 0;
}