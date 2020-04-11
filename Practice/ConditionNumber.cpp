#include "ConditionNumber.h"
#include "Utils.h"

double GetConditionNumber(const Matrix& m)
{
	return Utils::CubicNorm(m) * Utils::CubicNorm(InvMatrix(m));
}

Matrix InvMatrix(const Matrix& m)
{
	const int n = m.GetCols();
	Matrix A = m;
	Matrix B = Matrix::GetEye(n);

	// Upper
	for (int i = 0; i < n - 1; ++i)
	{
		for (int j = i + 1; j < n; ++j)
		{
			const double l = -A(j, i) / A(i, i);
			A.AddMultipliedRow(j, i, l);
			B.AddMultipliedRow(j, i, l);
		}
	}

	// Lower
	for (int i = n - 1; i >= 0; --i)
	{
		const double z = 1 / A(i, i);
		A.MultiplyRow(i, z);
		B.MultiplyRow(i, z);

		for (int j = i - 1; j >= 0; --j)
		{
			const double l = -A(j, i);
			A.AddMultipliedRow(j, i, l);
			B.AddMultipliedRow(j, i, l);
		}
	}

	return B;
}
