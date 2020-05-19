#include "Relaxation.h"

#include <iostream>

Vector SolveRelaxation(const Matrix& m, const Vector& b, double epsilon, double w)
{
    const int n = b.size();
    Vector x = b, prev(n);

    do
    {
        prev = x;
        for (int i = 0; i < n; ++i)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                    sum += m(i, j) * x[j];
            }
            x[i] = (1 - w) * x[i] + w * (b[i] - sum + m(i, i) * x[i]) / m(i, i);
        }
    }
    while (
        Utils::EuclideanNorm(x - prev) >= epsilon
    );
    return x;
}
