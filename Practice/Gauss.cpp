#include "Gauss.h"
#include "Matrix.h"

//TODO
//Some ?problems? with copying
Vector SolveGauss(Matrix m, Vector b)
{
	const int n = b.size();

	std::vector<int> indexes(n);
	// 0, 1, 2, 3...
	std::iota(indexes.begin(), indexes.end(), 0);

	for (int k = 0; k < n - 1; ++k)
	{
		// Getting max element
		auto [col, row] = FindMax(m, k + 1);

		m.SwapColumns(k, col);
		m.SwapRows(k, row);

		std::swap(b[row], b[k]);
		// Swapping  indexes
		std::swap(indexes[col], indexes[k]);

		for (int i = k + 1; i < n; ++i)
		{
			const double l = -m(i, k) / m(k, k);
			m.AddMultipliedRow(i, k, l);
			b[i] += l * b[k];
		}
	}

	auto shuffledAnswer = Utils::SolveTriangle(m, b);
	Vector answer(n);

	for (int i = 0; i < n; i++)
	{
		// Unwinding
		answer[indexes[i]] = shuffledAnswer[i];
	}

	return answer;
}

std::pair<int, int> FindMax(const Matrix& m, int start)
{
	const int n = m.GetCols();
	double max = m(start, start);
	std::pair<int, int> indexes = {start, start};

	for (int i = start; i < n; ++i)
	{
		for (int j = start; j < n; ++j)
		{
			const double curr = m(i, j);
			if (curr > max)
			{
				max = curr;
				indexes = {i, j};
			}
		}
	}
	return indexes;
}
