#pragma once

#include <numeric>

#include "Matrix.h"
#include "Utils.h"

std::pair<int, int> FindMax(const Matrix& m, int start);

Vector SolveGauss(Matrix m, Vector b);
