#include "Cholesky.h"


std::pair<Matrix, Vector> BuildCholesky(Matrix m)
{
	const int n = m.GetCols();
	Vector D(n);
	
	for (int i = 0; i < n; ++i)
	{
		double lsqrt;
		const double d = m(i, i);
		if (d > 0)
		{
			D[i] = 1;
			lsqrt = 1.0 / sqrt(d);
		}
		else
		{
			D[i] = -1;
			lsqrt = 1.0 / sqrt(-d);
		}

		m.MultiplyRowPart(i, lsqrt, i, n);
		for (int j = i + 1; j < n; ++j)
		{
			const double l = -m(j, i) / m(i, i);
			m.AddMultipliedRowPart(j, i, l, j, n);
		}
	}
	
	return { m, D };
}

Vector SolveCholesky(const Matrix& LT, const Vector& D, const Vector& b)
{
	const int n = D.size();

	// L * y = b
	// y = DL^T
	Vector y(n);
	for (int i = 0; i < n; ++i)
	{
		double sum = 0;
		for (int j = i - 1; j >= 0; --j)
		{
			sum += y[j] * LT(j, i);
		}
		y[i] = (b[i] - sum) / LT(i, i);
	}

	// y = D^-1 * y
	for (int i = 0; i < n; ++i)
	{
		y[i] /= D[i];
	}

	// L^T*x = D^-1 * y
	return Utils::SolveUpperTriangle(LT, y);
}



