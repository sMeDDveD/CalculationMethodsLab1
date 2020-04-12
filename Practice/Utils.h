#pragma once
#include "Matrix.h"

namespace Utils
{
	double CubicNorm(const Matrix& m);

	Vector SolveTriangle(const Matrix& m, const Vector& b);
}
