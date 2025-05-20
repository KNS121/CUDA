#pragma once

#ifndef MATRIXMULCPU_H
#define MATRIXMULCPU_H

#include <vector>
using std::vector;

vector<vector<int>> MatrixMultiplyCPU(const vector<vector<int>>& A,
    const vector<vector<int>>& B, const int N);

#endif
