#include "LeastSquares.h"

#include <iostream>

Vector SolveLeastSquares(Matrix m, Vector b)
{
	GaussTransform(m, b);
	
	auto [LT, D] = BuildCholesky(m);
	return SolveCholesky(LT, D, b);
}

void GaussTransform(Matrix& m, Vector& b)
{
	Matrix T = m.Transpose();
	m = T * m;
	b = T * b;
}
