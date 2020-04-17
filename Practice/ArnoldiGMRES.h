#pragma once

#include "Matrix.h"
#include "Utils.h"
#include "LeastSquares.h"

Vector SolveArnoldiGMRES(Matrix m, Vector b, double epsilon);