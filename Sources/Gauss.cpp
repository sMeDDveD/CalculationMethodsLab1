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
        auto [col, row] = Utils::FindMax(m, k + 1);

        m.SwapColumns(k, col);
        m.SwapRows(k, row);

        std::swap(b[row], b[k]);
        // Swapping  indexes
        std::swap(indexes[col], indexes[k]);

        for (int i = k + 1; i < n; ++i)
        {
            const double l = -m(i, k) / m(k, k);
            m.AddMultipliedRowPart(i, k, l, k + 1, n);
            b[i] += l * b[k];
        }
    }

    auto shuffledAnswer = Utils::SolveUpperTriangle(m, b);
    Vector answer(n);

    for (int i = 0; i < n; i++)
    {
        // Unwinding
        answer[indexes[i]] = shuffledAnswer[i];
    }

    return answer;
}
