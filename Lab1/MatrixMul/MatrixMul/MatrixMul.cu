#include "MatrixMulCPU.h"
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <iostream>

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

int main() {

    vector<vector<int>> A(N, vector<int>(N));

    fillMatrix(A);

    printMatrix(A);

    return 0;
}