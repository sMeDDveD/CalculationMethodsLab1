#include <iostream>
#include <functional>

#include "ConditionNumber.h"
#include "Matrix.h"
#include "Utils.h"
#include "Gauss.h"
#include "LUP.h"
#include "Cholesky.h"

using namespace std;

void test(const std::function<Vector(Matrix, Vector)> solver, Matrix m, Vector b, Vector answer)
{
	auto c = solver(m, b);
	for (auto x : c)
	{
		std::cout << x << " ";
	}

	std::cout << endl;

	for (int i = 0; i < answer.size(); i++)
	{
		std::cout << abs(c[i] - answer[i]) << " ";
	}

	std::cout << endl;
}

int main()
{
	double upper[] = {1, 2, 3, 0, 1, -6, 0, 0, -48};
	double lower[] = { 3, 0, 0, 1, 1, 0, 1, 2, 3 };
	double arr[] = {10, 1, 2, 1, 6, 3, 2, 3, 7};//{1, 2, 3, 4, 9, 6, 7, 8, 9};
	Matrix lowerTriagonal = Matrix::FromArray(
		lower, 3, 3
	);
	Matrix upperTriagonal = Matrix::FromArray(
		upper, 3, 3
	);
	Matrix full = Matrix::FromArray(
		arr, 3, 3
	);
	Vector b = { -2, -2, -2 };

	auto [LT, D] = BuildCholesky(full);
	auto [LU, P] = BuildLUP(full);
	auto x = SolveCholesky(LT, D, b);
	SolveLUP(LU, P, b);
	Vector answer = {1, 0, -1};
	test(SolveGauss, full, b, answer);
	test(Utils::SolveLowerTriangle, lowerTriagonal, { 3, 1, -2 }, answer);
	system("pause");
	return 0;
}
