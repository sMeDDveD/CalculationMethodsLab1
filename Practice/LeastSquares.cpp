#include "LeastSquares.h"

#include <iostream>

Vector SolveLeastSquares(Matrix m, Vector b, bool QR)
{
    if (!QR)
    {
        GaussTransform(m, b);

        auto[LT, D] = BuildCholesky(m);
        return SolveCholesky(LT, D, b);
    }
    else
    {
        return SolveHouseholder(m, b);
    }
}

    
void GaussTransform(Matrix& m, Vector& b)
{
    Matrix T = m.Transpose();
    m = T * m;
    b = T * b;
}
