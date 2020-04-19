#include "Utils.h"

Vector operator*(const Vector& l, double lambda)
{
	Vector r = l;
	for (auto& now : r)
	{
		now *= lambda;
	}
	return r;
}

Vector operator/(const Vector& l, double lambda)
{
	return l * (1 / lambda);
}

Vector operator*(double lambda, const Vector& l)
{
	return l * lambda;
}

Vector operator+(const Vector& l, const Vector& r)
{
	const int n = l.size();

	Vector x(n);
	for (int i = 0; i < n; ++i)
	{
		x[i] = l[i] + r[i];
	}

	return x;
}

Vector operator-(const Vector& l, const Vector& r)
{
	const int n = l.size();

	Vector x(n);
	for (int i = 0; i < n; ++i)
	{
		x[i] = l[i] - r[i];
	}

	return x;
}

double Utils::CubicNorm(const Matrix& m)
{
    double norm = 0;
    for (int i = 0; i < m.GetRows(); ++i)
    {
        double sum = 0;
        for (int j = 0; j < m.GetCols(); ++j)
        {
            sum += std::abs(m(i, j));
        }
        norm = std::max(norm, sum);
    }
    return norm;
}

double Utils::EuclideanNorm(const Vector& v)
{
    return sqrt(ScalarMultiply(v, v));
}


double Utils::ScalarMultiply(const Vector& l, const Vector& r)
{
    double sum = 0;
    for (int i = 0; i < l.size(); ++i)
    {
        sum += l[i] * r[i];
    }
    return sum;
}


Vector Utils::SubVectors(const Vector& l, const Vector& r)
{
	return l - r;
}

Vector Utils::SolveUpperTriangle(const Matrix& m, const Vector& b)
{
    const int n = m.GetCols();
    Vector x(n);

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
    const int n = m.GetCols();
    Vector x(n);

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
    double max = std::abs(m(start, start));
    std::pair<int, int> indexes = {start, start};

    for (int i = start; i < n; ++i)
    {
        for (int j = start; j < n; ++j)
        {
            const double curr = std::abs(m(i, j));
            if (curr > max)
            {
                max = curr;
                indexes = {i, j};
            }
        }
    }
    return indexes;
}

double Utils::CubicNorm(const Vector &v)
{
    double m = v[0];
    for (const auto &now : v)
    {
        m = std::max(std::abs(now), m);
    }
    return m;
}

Vector Utils::GenerateVector(int length, int n)
{
    std::random_device device;
    std::uniform_real_distribution<double> distr(
            -pow(2, static_cast<double>(n) / 4),
            pow(2, static_cast<double>(n) / 4)
    );

    Vector v(length);

    for (auto &now : v)
    {
        now = distr(device);
    }

    return v;
}
