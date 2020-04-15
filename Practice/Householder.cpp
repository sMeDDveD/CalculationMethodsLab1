#include <iostream>
#include "Householder.h"


static Vector GetW(Vector a)
{
    const int n = a.size();
    Vector r(n, 0);
    r[0] = -Utils::sgn(a[0]) * Utils::EuclideanNorm(a);

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
    const int vectorSize = w.size();
    const int offset = m.GetRows() - vectorSize;
    double scalar;

    for (int j = 0; j < n; ++j)
    {
        scalar = 0;
        for (int k = 0; k < vectorSize; ++k)
        {
            scalar += m(k + offset, j) * w[k];
        }

        for (int i = 0; i < vectorSize; ++i)
        {
            m(i + offset, j) -= 2 * scalar * w[i];
        }
    }

    scalar = 0;
    for (int k = 0; k < vectorSize; ++k)
    {
        scalar += b[k + offset] * w[k];
    }

    for (int i = 0; i < vectorSize; ++i)
    {
        b[i + offset] -= 2 * scalar * w[i];
    }
}

Vector SolveHouseholder(Matrix m, Vector b)
{
    const int n = m.GetCols();
    const int l = m.GetRows();
    for (int i = 0; i < n - 1; ++i)
    {
        ApplyToMatrix(m, b, GetW(m.GetColPart(i, i, l)));
    }


    return Utils::SolveUpperTriangle(m, b);
}
