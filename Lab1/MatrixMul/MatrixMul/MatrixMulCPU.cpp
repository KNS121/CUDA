#include "MatrixMulCPU.h"
#include <vector>
#include <stdexcept>

using namespace std;
using std::vector;

vector<vector<int>> MatrixMultiplyCPU(const vector<vector<int>>& A, const vector<vector<int>>& B, const int N){

    vector<vector<int>> resultMatrix(N, vector<int>(N, 0));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < N; ++k) {
                resultMatrix[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return resultMatrix;
}