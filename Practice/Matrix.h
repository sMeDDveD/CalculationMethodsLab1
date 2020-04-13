#pragma once
#include <exception>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>

using Vector = std::vector<double>;

class Matrix final
{
	double* data = nullptr;
	int rows, cols;

public:
	const static int precision = 3;
	const static size_t variant = 9;

	static Matrix FromArray(double* data, int rows, int cols);

	explicit Matrix(int rows, int cols);
	explicit Matrix(int n);


	int GetRows() const;
	int GetCols() const;
	
	Vector GetRow(int row) const;
	Vector GetCol(int col) const;
	Vector GetRowPart(int row, int start, int end) const;
	Vector GetColPart(int col, int start, int end) const;

	Matrix(Matrix&& other) noexcept;
	Matrix& operator=(Matrix&& other) noexcept;

	Matrix& operator=(const Matrix& other);

	Matrix(const Matrix& other);

	[[nodiscard]] double* GetData() const;
	double operator()(int i, int j) const;
	double& operator()(int i, int j);

	Matrix operator*(const Matrix& other) const;
	Vector operator*(const Vector& other) const;
	
	Matrix operator+(const Matrix& other) const;

	Matrix GetSubMatrix(int i, int j) const;
	void SwapRows(int fRow, int sRow);
	void SwapColumns(int fCol, int sCol);

	void AddMultipliedRow(int to, int from, double lambda);
	void AddMultipliedRowPart(int to, int from, double lambda, int start, int end);
	
	void MultiplyRow(int row, double lambda);
	void MultiplyRowPart(int row, double lambda, int start, int end);

	Matrix Transpose() const;

	static Matrix GetEmpty(int n, int m);
	static Matrix GetEye(int n);
	static Matrix GenerateMatrix(int n, int param);

	~Matrix();

	friend std::ostream& operator<<(std::ostream& out, const Matrix& matrix);
};
