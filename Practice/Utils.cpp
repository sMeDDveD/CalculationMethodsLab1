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
