#pragma once

#include <utility>

#include "Matrix.h"
#include "Utils.h"


std::pair<Matrix, Vector> BuildCholesky(Matrix m);

Vector SolveCholesky(const Matrix& LT, const Vector& D, const Vector& b);
