#include <iostream>
#include <functional>

#include "ConditionNumber.h"
#include "Matrix.h"
#include "Utils.h"
#include "Gauss.h"
#include "LUP.h"
#include "Cholesky.h"
#include "Relaxation.h"
#include "Householder.h"
#include "LeastSquares.h"

void tests(const Matrix A, const Vector b, const Vector x)
{
	std::cout << "Gauss: " << std::endl;
	std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveGauss(A, b)));
	std::cout << std::endl;

	std::cout << "LUP: " << std::endl;
	auto[LU, P] = BuildLUP(A);
	std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveLUP(LU, P, b)));
	std::cout << std::endl;

	std::cout << "Cholesky:" << std::endl;
	auto[LT, D] = BuildCholesky(A);
	std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveCholesky(LT, D, b)));
	std::cout << std::endl;
}


int main()
{
	double arr[] = {10, 1, 2, 1, 6, 3, 2, 3, 7};
	Matrix A = Matrix::FromArray(
		arr, 3, 3
	);
	Vector x = { 1, 2 , 3 };
	Vector b = A * x;

	tests(A, b, x);
	system("pause");
	return 0;
}
