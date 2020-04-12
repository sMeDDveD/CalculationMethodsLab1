#include <iostream>
#include <functional>

#include "ConditionNumber.h"
#include "Matrix.h"
#include "Utils.h"
#include "Gauss.h"

using namespace std;

void test(std::function<Vector(Matrix, Vector)> solver, Matrix m, Vector b, Vector answer)
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
	double triag[] = { 1, 2, 3, 0, 1, -6, 0, 0, -48 };
	double arr[] = { 1, 2, 3, 4, 9, 6, 7, 8, 9 };
	Matrix upperTriagonal = Matrix::FromArray(
		triag, 3, 3
	);
	Matrix full = Matrix::FromArray(
		arr, 3, 3
	);
	Vector b = { -2, -2, -2 };
	Vector answer = { 1, 0, -1 };
	test(SolveGauss, full, b, answer);
	system("pause");
	return 0;
}