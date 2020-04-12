#include "Utils.h"

#include <iostream>

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

Vector Utils::SolveUpperTriangle(const Matrix& m, const Vector& b)
{
	const int n = b.size();
	Vector x(b.size());

	for (int i = n - 1; i >= 0; --i)
	{
		double sum = 0;
		for (int j = i + 1; j < n; ++j)
		{
			sum += x[j] * m(i, j);
		}
		x[i] = (b[i] - sum) / m(i, i);
	}

	return x;
}

Vector Utils::SolveLowerTriangle(const Matrix& m, const Vector& b)
{
	const int n = b.size();
	Vector x(b.size());
	for (int i = 0; i < n; ++i)
	{
		double sum = 0;
		for (int j = i - 1; j >= 0; --j)
		{
			sum += x[j] * m(i, j);
		}
		x[i] = (b[i] - sum) / m(i, i);
	}
	
	return x;
}

std::pair<int, int> Utils::FindMax(const Matrix& m, int start)
{
	const int n = m.GetCols();
	double max = abs(m(start, start));
	std::pair<int, int> indexes = { start, start };

	for (int i = start; i < n; ++i)
	{
		for (int j = start; j < n; ++j)
		{
			const double curr = abs(m(i, j));
			if (curr > max)
			{
				max = curr;
				indexes = { i, j };
			}
		}
	}
	return indexes;
}
