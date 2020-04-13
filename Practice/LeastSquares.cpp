#include "LeastSquares.h"

#include <iostream>

Vector SolveLeastSquares(Matrix m, Vector b)
{
	GaussTransform(m, b);
	
	auto [LT, D] = BuildCholesky(m);
	std::cout << m;
	std::cout << LT;
	return SolveCholesky(LT, D, b);
}

void GaussTransform(Matrix& m, Vector& b)
{
	Matrix T = m.Transpose();
	std::cout << T;
	m = T * m;
	b = T * b;
}