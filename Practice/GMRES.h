#pragma once

#include "Matrix.h"
#include "LeastSquares.h"
#include "Utils.h"

Vector SolveGMRES(Matrix m, Vector b, double epsilon);

Matrix Stack(const std::vector<Vector>& v);