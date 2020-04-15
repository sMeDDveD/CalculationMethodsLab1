#pragma once

#include "Matrix.h"
#include "Utils.h"
#include "Cholesky.h"
#include "Householder.h"

Vector SolveLeastSquares(Matrix m, Vector b, bool QR = false);

void GaussTransform(Matrix& m, Vector& b);
