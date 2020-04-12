#pragma once

#include "Matrix.h"
#include "Utils.h"

Vector SolveRelaxation(const Matrix& m, const Vector& b, double epsilon, double w);
