#include "Utils.h"

double Utils::CubicNorm(const Matrix& m)
{
	double norm = 0;
	for (int i = 0; i < m.GetRows(); ++i)
	{
		double sum = 0;
		for (int j = 0; j < m.GetCols(); ++j)
		{
			sum += abs(m(i, j));
		}
		norm = std::max(norm, sum);
	}
	return norm;
}

Vector Utils::SolveTriangle(const Matrix& m, const Vector& b)
{
	const int n = b.size();
	Vector x(b.size());

	for (int i = n - 1; i >= 0; i--)
	{
		double sum = 0;
		for(int j = i + 1; j < n; j++)
		{
			sum += x[j] * m(i, j);
		}
		x[i] = (b[i] - sum) / m(i, i);
	}

	return x;
}
