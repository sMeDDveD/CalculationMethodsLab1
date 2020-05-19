#include "ConditionNumber.h"
#include "Utils.h"

double GetConditionNumber(const Matrix& m)
{
    return Utils::CubicNorm(m) * Utils::CubicNorm(InvMatrix(m));
}

Matrix InvMatrix(Matrix m)
{
    const int n = m.GetCols();
    Matrix B = Matrix::GetEye(n);

    // Upper
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            const double l = -m(j, i) / m(i, i);
            m.AddMultipliedRow(j, i, l);
            B.AddMultipliedRow(j, i, l);
        }
    }

    // Lower
    for (int i = n - 1; i >= 0; --i)
    {
        const double z = 1 / m(i, i);
        m.MultiplyRow(i, z);
        B.MultiplyRow(i, z);

        for (int j = i - 1; j >= 0; --j)
        {
            const double l = -m(j, i);
            m.AddMultipliedRow(j, i, l);
            B.AddMultipliedRow(j, i, l);
        }
    }

    return B;
}
