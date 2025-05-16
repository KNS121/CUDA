#include "MatrixMulCPU.h"
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

using namespace std;
using std::vector;
using namespace std::chrono;


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


__global__ void MatrixMultplyGPU(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}



vector<vector<int>> MatrixMultCUDA(const vector<vector<int>>& A, const vector<vector<int>>& B, const int n) {

    //one dim arrays
    int* one_dim_array_A = new int[n * n];
    int* one_dim_array_B = new int[n * n];
    int* one_dim_array_reuslt = new int[n * n];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            one_dim_array_A[i * n + j] = A[i][j];
            one_dim_array_B[i * n + j] = B[i][j];
        }
    }


    // go to CUDA
    int* dev_A;
    int* dev_B;
    int* dev_res;

    cudaMalloc(&dev_A, n * n * sizeof(int));
    cudaMalloc(&dev_B, n * n * sizeof(int));
    cudaMalloc(&dev_res, n * n * sizeof(int));

    cudaMemcpy(dev_A, one_dim_array_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, one_dim_array_B, n * n * sizeof(int), cudaMemcpyHostToDevice);

    MatrixMultplyGPU << <1, 1 >> > (dev_A, dev_B, dev_res, n);

    // obratno
    cudaMemcpy(one_dim_array_reuslt, dev_res, n * n * sizeof(int), cudaMemcpyDeviceToHost);
   
    vector<vector<int>> resultMatrix(n, vector<int>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            resultMatrix[i][j] = one_dim_array_reuslt[i * n + j];
        }
    }

    // pochistim

    delete[] one_dim_array_A;
    delete[] one_dim_array_B;
    delete[] one_dim_array_reuslt;

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_res);

    
    return resultMatrix;
}


int main() {

    const int n = 1;

    vector<vector<int>> A(n, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(n));

    fillMatrix(A);
    fillMatrix(B);

    auto start_cpu = chrono::high_resolution_clock::now();
    vector<vector<int>> res_from_CPU = MatrixMultiplyCPU(A, B, n);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_time = end_cpu - start_cpu;

    auto start_gpu = chrono::high_resolution_clock::now();
    vector<vector<int>> res_from_GPU = MatrixMultCUDA(A, B, n);
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;


    cout << "GPU time: " << gpu_time.count() << " secund \n";
    cout << "GPU result:\n";
    printMatrix(res_from_GPU);
    
    cout << "CPU time: " << cpu_time.count() << " secund \n";
    cout << "CPU result:\n";
    printMatrix(res_from_CPU);

    return 0;
}