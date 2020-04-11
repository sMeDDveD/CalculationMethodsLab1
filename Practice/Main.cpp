#include <iostream>

#include "ConditionNumber.h"
#include "Matrix.h"

using namespace std;

int main()
{
	double arr[] = { 1, 2, 3, 4, 9, 6, 7, 8, 9 };
	Matrix E = Matrix::GetEye(3);
	Matrix matrix = Matrix::FromArray(arr, 3, 3);
	cout << Matrix::GenerateMatrix(4, 5) << std::endl;
	cout << InvMatrix(matrix);
	cout << matrix << std::endl;
	system("pause");
	return 0;
}