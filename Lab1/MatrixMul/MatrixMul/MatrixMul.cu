#include "MatrixMulCPU.h"
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
using namespace std;
using std::vector;

#define N 5

void fillMatrix(vector<vector<int>>& matrix) {
    for (auto& row : matrix) {
        for (auto& element : row) {
            element = rand() % 100;
        }
    }
}


void printMatrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            std::cout << element << "\t";
        }
        std::cout << std::endl;
    }
}


//__global__ void MatrixMultiplyGPU(const int *a, )


__global__ void check_sum(int* a, int* b, int* res) {
    *res = *a + *b;
}



int main() {

    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));

    fillMatrix(A);
    fillMatrix(B);


    int a = 1;
    int b = 2;
    int res;

    int* dev_a;
    int* dev_b; 
    int* dev_res;

    cudaMalloc((void**)&dev_a, sizeof(int));
    cudaMalloc((void**)&dev_b, sizeof(int));
    cudaMalloc((void**)&dev_res, sizeof(int));

    cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    check_sum << <1, 1 >> > (dev_a, dev_b, dev_res);

    cudaMemcpy(&res, dev_res, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d + %d = %d\n", a, b, res);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);


    return 0;
}