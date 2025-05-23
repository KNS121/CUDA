#include "MatrixMulCPU.h"
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <cuda_fp16.h>

#include <cooperative_groups.h>

#define BLOCK_SIZE 32
#define WARPSIZE 32

using namespace cooperative_groups;

#define N 1024
#define BLOCK_SIZE 32

dim3 kolvo_potokov(32, 32);
dim3 kolvo_blockov(32, 32);

using namespace std;
using std::vector;
using namespace std::chrono;


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


bool proverka_results(const vector<vector<int>>& Res_CPU, const vector<vector<int>>& Res_GPU, const int n) {


    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Res_GPU[i][j] != Res_CPU[i][j]) {
                return false;
            }
        }
    }
    return true;
}




__global__ void MatrixMultplyGPU_WaprReduce(const int* A, const int* B, int* C, int n) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
    int lane = threadIdx.x % WARPSIZE;

    if (warpId >= n * n) return;

    int row = warpId / n;
    int col = warpId % n;

    int partial_sum = 0;

    // patricle of sum
    for (int k = lane; k < n; k += WARPSIZE) {
        int a_val = A[row * n + k];
        int b_val = B[k * n + col];
        partial_sum += a_val * b_val;
    }

    // reduce
    unsigned mask = 0xffffffff;
    for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // zapis' result with 0 thread
    if (lane == 0) {
        C[row * n + col] = partial_sum;
    }
}



__global__ void MatrixMultplyGPU_Wapr_withShared(const int* a, const int* b, int* c, int n) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    int idx = threadIdx.x;
    int idy = threadIdx.y;


    __shared__ int array_a_in_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int array_b_in_shared[BLOCK_SIZE][BLOCK_SIZE];


    float blocks_pokrytie = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;


    int res = 0;



    for (int i = 0; i < blocks_pokrytie; ++i) {

        int index_of_A = (idx + i * BLOCK_SIZE);
        int index_of_B = i * BLOCK_SIZE + idy;

        if (row < n && index_of_A < n) {

            array_a_in_shared[idy][idx] = a[row * n + index_of_A];
        }
        else { array_a_in_shared[idy][idx] = 0; }

        if (col < n && index_of_B < n) {
            array_b_in_shared[idx][idy] = b[index_of_B * n + col];
        }

        else { array_b_in_shared[idx][idy] = 0; }



        __syncthreads();


        //+group warp
        coalesced_group warp = coalesced_threads();
        int thread_element_a = array_a_in_shared[idy][idx];


        for (int lane = 0; lane < BLOCK_SIZE; ++lane) {
            int a_val_from_other_threads = warp.shfl(thread_element_a, lane);
            res += a_val_from_other_threads * array_b_in_shared[idx][lane];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = res;
    }
}






vector<vector<int>> MatrixMultCUDA_warp_with_reduce(const vector<vector<int>>& A, const vector<vector<int>>& B, const int n) {

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


    int total_warps = (n * n);

    int threads_per_block = 256;
    int blocks = (total_warps * WARPSIZE + threads_per_block - 1) / threads_per_block;

    MatrixMultplyGPU_WaprReduce << <blocks, threads_per_block >> > (dev_A, dev_B, dev_res, n);

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



vector<vector<int>> MatrixMultCUDA_warp_shared(const vector<vector<int>>& A, const vector<vector<int>>& B, const int n) {

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

    MatrixMultplyGPU_Wapr_withShared << < kolvo_blockov, kolvo_potokov >> > (dev_A, dev_B, dev_res, n);

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

    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));



    fillMatrix(A);
    fillMatrix(B);

    // cout << "Matrix A\n" << endl;
    // printMatrix(A);
    // cout << "\n" << endl;
    // cout << "Matrix B\n" << endl;
    // printMatrix(B);

    // cout << "\n" << endl;

    auto start_cpu = chrono::high_resolution_clock::now();
    vector<vector<int>> res_from_CPU = MatrixMultiplyCPU(A, B, N);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_time = end_cpu - start_cpu;

    auto start_gpu_with_shared = chrono::high_resolution_clock::now();
    vector<vector<int>> res_from_GPU_with_shared = MatrixMultCUDA_warp_shared(A, B, N);
    auto end_gpu_with_shared = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time_with_shared = end_gpu_with_shared - start_gpu_with_shared;

    auto start_gpu_warp_with_reduce = chrono::high_resolution_clock::now();
    vector<vector<int>> res_from_GPU_with_reduce = MatrixMultCUDA_warp_with_reduce(A, B, N);
    auto end_gpu_warp_with_reduce = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time_warp_with_reduce = end_gpu_warp_with_reduce - start_gpu_warp_with_reduce;


    bool check_with_shared = proverka_results(res_from_CPU, res_from_GPU_with_shared, N);

    bool check_with_reduce = proverka_results(res_from_CPU, res_from_GPU_with_reduce, N);

    cout << "Razmer matrix : " << N << " \n";

    cout << "GPU time ( Warp with shared ) : " << gpu_time_with_shared.count() << " secund \n";
    cout << "GPU time ( Warp with reduce ) : " << gpu_time_warp_with_reduce.count() << " secund \n";
    //cout << "GPU result:\n";
    //printMatrix(res_from_GPU);

    cout << "CPU time: " << cpu_time.count() << " secund \n";
    //cout << "CPU result:\n";
    //printMatrix(res_from_CPU);



    cout << "proverka rezov  ( Warp + shared ) - " << check_with_shared << "\n";

    cout << "proverka rezov  ( Warp + reduce ) - " << check_with_reduce << "\n";

    return 0;
}