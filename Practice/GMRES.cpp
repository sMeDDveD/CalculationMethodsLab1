#include <iostream>
#include "GMRES.h"

Vector SolveGMRES(Matrix m, Vector b, double epsilon)
{
    const int n = b.size();
    std::vector<Vector> k;
    k.push_back(b);

    Vector x;
    for (int i = 0; i < n; ++i) {
        auto K = Stack(k);
        x = K * SolveLeastSquares(m * K, b);
        std::cout << x.size();
        k.push_back(m * k.back());
        if (Utils::EuclideanNorm(Utils::SubVectors(b, m * x)) < epsilon) {
            return x;
        }
    }

    return x;
}



Matrix Stack(const std::vector<Vector> &v)
{
    const int rows = v[0].size();
    const int cols = v.size();
    Matrix m(rows, cols);

    for(int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            m(i, j) = v[j][i];
        }
    }

    return m;
}
