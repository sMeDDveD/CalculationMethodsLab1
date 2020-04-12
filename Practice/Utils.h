#pragma once
#include "Matrix.h"

namespace Utils
{
	double CubicNorm(const Matrix& m);

	double EuclideanNorm(const Vector& v);

	double ScalarMultiply(const Vector& l, const Vector& r);

	Vector SubVectors(const Vector& l, const Vector& r);

	Vector SolveUpperTriangle(const Matrix& m, const Vector& b);

	Vector SolveLowerTriangle(const Matrix& m, const Vector& b);

	std::pair<int, int> FindMax(const Matrix& m, int start);
}
