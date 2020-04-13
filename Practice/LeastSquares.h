#pragma once

#include "Matrix.h"
#include "Utils.h"
#include "Cholesky.h"

Vector SolveLeastSquares(Matrix m, Vector b);

void GaussTransform(Matrix& m, Vector& b);
