#include "ArnoldiGMRES.h"

#include <iostream>


static Matrix HessenbergStack(const std::vector<Vector>& v, int k)
{
	Matrix m(k + 1, k);

	for (int i = 0; i < k + 1; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
            // TODO
			m(i, j) = i < v[j].size() ? v[j][i] : 0;
		}
	}

	return m;
}


Vector SolveArnoldiGMRES(Matrix m, Vector b, double epsilon)
{
	const int n = b.size();

	std::vector<Vector> Q;
	std::vector<Vector> H;
	Vector x(n);

    const double bNorm = Utils::EuclideanNorm(b);

	Q.push_back(b / bNorm);
	Vector d(n + 1);
	d[0] = bNorm;
    
    for (int j = 0, k = 1; j < k; ++j)
    {
		Vector z = m * Q[j];

		H.emplace_back(j + 2);
        for (int i = 0; i <= j; ++i)
        {
			H[j][i] = Utils::ScalarMultiply(z, Q[i]);
			z = Utils::SubVectors(z, H[j][i] * Q[i]);
        }
        const auto h = Utils::EuclideanNorm(z);

		H[j][j + 1] = h;


        //TODO: Speedup with Givens and fast mul
		auto y = SolveLeastSquares(HessenbergStack(H, j + 1), d, true);
		x = Stack(Q) * y;

        if (Utils::EuclideanNorm(Utils::SubVectors(b, m * x)) < epsilon)
        {
			return x;
        }

        
		Q.push_back(z / h);
		++k;
    }

	return x;
}