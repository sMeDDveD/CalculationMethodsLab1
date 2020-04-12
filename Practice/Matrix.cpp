#include "Matrix.h"

#include <iomanip>
#include <random>

Matrix Matrix::FromArray(double* data, int rows, int cols)
{
	Matrix matrix(rows, cols);
	std::copy(data, data + rows * cols, matrix.data);

	return matrix;
}

Matrix::Matrix(int rows, int cols): rows(rows), cols(cols)
{
	data = new double[rows * cols];
}

Matrix::Matrix(int n): Matrix(n, n)
{
}

int Matrix::GetRows() const
{
	return rows;
}

int Matrix::GetCols() const
{
	return cols;
}

Matrix::Matrix(Matrix&& other) noexcept
{
	rows = other.rows;
	cols = other.cols;
	std::swap(data, other.data);
}

Matrix& Matrix::operator=(const Matrix& other)
{
	if (this == &other)
		return *this;

	rows = other.rows;
	cols = other.cols;

	data = new double[rows * cols];
	std::copy(other.data, other.data + rows * cols, data);

	return *this;
}

Matrix::Matrix(const Matrix& other): rows(other.rows),
                                     cols(other.cols)
{
	this->operator=(other);
}

double* Matrix::GetData() const
{
	double* copy = new double[rows * cols];
	std::copy(data, data + rows * cols, copy);

	return copy;
}

double& Matrix::operator()(int i, int j)
{
	if (i >= rows || j >= cols)
		throw std::exception("Matrix subscript out of bounds");
	return data[i * cols + j];
}

Matrix Matrix::operator*(const Matrix& other) const
{
	Matrix product = Matrix(rows, other.cols);

	if (cols != other.rows)
	{
		throw std::exception("Unsupported sizes");
	}

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < other.cols; j++)
		{
			double ij = 0;
			for (int k = 0; k < cols; k++)
			{
				ij += this->operator()(i, k) * other(k, j);
			}
			product(i, j) = ij;
		}
	}

	return product;
}

Matrix Matrix::operator+(const Matrix& other) const
{
	if (other.cols != cols || other.rows != rows)
	{
		throw std::exception("Unsupported sizes");
	}

	double* aData = new double[rows * cols];
	for (int i = 0; i < rows * cols; ++i)
	{
		aData[i] = data[i] + other.data[i];
	}

	return FromArray(aData, rows, cols);
}

void Matrix::SwapRows(int fRow, int sRow)
{
	for (int j = 0; j < rows; ++j)
	{
		std::swap(this->operator()(fRow, j), this->operator()(sRow, j));
	}
}

void Matrix::SwapColumns(int fCol, int sCol)
{
	for (int i = 0; i < cols; ++i)
	{
		std::swap(this->operator()(i, fCol), this->operator()(i, sCol));
	}
}

void Matrix::AddMultipliedRow(int to, int from, double lambda)
{
	AddMultipliedRowPart(to, from, lambda, 0, cols);
}

void Matrix::AddMultipliedRowPart(int to, int from, double lambda, int start, int end)
{
	for (int j = start; j < end; ++j)
	{
		this->operator()(to, j) += lambda * this->operator()(from, j);
	}
}

void Matrix::MultiplyRow(int row, double lambda)
{
	MultiplyRowPart(row, lambda, 0, cols);
}

void Matrix::MultiplyRowPart(int row, double lambda, int start, int end)
{
	for (int j = start; j < end; ++j)
	{
		this->operator()(row, j) *= lambda;
	}
}

Matrix Matrix::GetEmpty(int n, int m)
{
	return FromArray(new double[n * m](), n, m);
}

Matrix Matrix::GetEye(int n)
{
	Matrix m = GetEmpty(n, n);
	for (int i = 0; i < n; ++i)
	{
		m(i, i) = 1;
	}
	return m;
}

Matrix Matrix::GenerateMatrix(int n, int param)
{
	Matrix m(n, n);

	std::random_device device;
	const std::uniform_real_distribution<double> distr(
		-pow(2, static_cast<double>(param) / 4),
		pow(2, static_cast<double>(param) / 4)
	);

	for (int i = 0; i < n; ++i)
	{
		for (int j = i + 1; j < n; j++)
		{
			const auto generated = distr(device);
			m(i, j) = m(j, i) = generated;
		}

		double d = 0;
		m(i, i) = 0;
		for (int j = 0; j < n; j++)
		{
			d += abs(m(i, j));
		}
		m(i, i) = d;
	}

	return m;
}

double Matrix::operator()(int i, int j) const
{
	if (i >= rows || j >= cols)
		throw std::exception("Matrix subscript out of bounds");
	return data[i * cols + j];
}

Matrix::~Matrix()
{
	delete[] data;
}

std::ostream& operator<<(std::ostream& out, const Matrix& matrix)
{
	out << std::setw(8) << std::setprecision(4);
	for (int i = 0; i < matrix.rows; i++)
	{
		for (int j = 0; j < matrix.cols; j++)
		{
			out << std::setw(8) << matrix(i, j) << ' ';
		}
		out << std::endl;
	}

	return out;
}
