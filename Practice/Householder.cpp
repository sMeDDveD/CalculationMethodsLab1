#include "Householder.h"

#include <iostream>


static Vector GetW(Vector a)
{
	const int n = a.size();
	Vector r(n, 0); r[0] = -Utils::sgn(a[0]) * Utils::EuclideanNorm(a);

	Vector w = Utils::SubVectors(a, r);
	const double norm = Utils::EuclideanNorm(w);

	for (auto& now : w)
	{
		now /= norm;
	}

	return w;
}

static void ApplyToMatrix(Matrix& m, Vector& b, const Vector& w)
{
	const int n = m.GetCols();
	double scalar = 0;
	
	for (int j = 0; j < n; ++j)
	{
		for (int k = 0; k < n; ++k)
		{
			scalar += m(k, j) * w[k];
		}

		for (int i = 0; i < n; ++i)
		{
			m(i, j) -= 2 * scalar * w[i];
		}
	}

	for (int k = 0; k < n; ++k)
	{
		scalar += b[k] * w[k];
	}

	for (int i = 0; i < n; ++i)
	{
		b[i] -= 2 * scalar * w[i];
	}

	std::cout << m;
}

Vector SolveHouseholder(Matrix m, Vector b)
{
	
	return b;
}
