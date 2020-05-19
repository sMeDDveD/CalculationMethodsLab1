#include <iostream>
#include <functional>
#include <chrono>
#include <utility>
#include <fstream>

#include "ConditionNumber.h"
#include "Matrix.h"
#include "Utils.h"
#include "Gauss.h"
#include "LUP.h"
#include "Cholesky.h"
#include "Relaxation.h"
#include "Householder.h"
#include "LeastSquares.h"
#include "GMRES.h"
#include "ArnoldiGMRES.h"

constexpr double EPS = 0.00000001;
constexpr double W = 10.0 / 6;

template < class T >
std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
	os << "x = (";
	for (auto ii = v.begin(); ii != v.end() - 1; ++ii)
	{
		os << *ii << ", ";
	}
	os << v.back() << ")";
	return os;
}

std::pair<double, double> task2(Matrix A) {
    auto c = GetConditionNumber(A);

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    start = std::chrono::high_resolution_clock::now();
    InvMatrix(std::move(A));
    end = std::chrono::high_resolution_clock::now();

    auto t = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    return {c, t};
}

std::pair<double, double> task3(Matrix A, Vector b, const Vector& x) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    start = std::chrono::high_resolution_clock::now();
    auto norm = Utils::CubicNorm(x - SolveGauss(std::move(A), std::move(b)));
    end = std::chrono::high_resolution_clock::now();

    auto t = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    return {norm, t};
}

std::tuple<double, double, double> task4(Matrix A, const Vector& b, const Vector& x) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    auto [LU, P] = BuildLUP(std::move(A));
    end = std::chrono::high_resolution_clock::now();

    auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    start = std::chrono::high_resolution_clock::now();
    auto norm = Utils::CubicNorm(x - SolveLUP(LU, P, b));
    end = std::chrono::high_resolution_clock::now();

    auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    return {t1, t2, norm};
}

std::tuple<double, double, double> task5(Matrix A, const Vector& b, const Vector& x) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    auto [LT, D] = BuildCholesky(std::move(A));
    end = std::chrono::high_resolution_clock::now();

    auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    start = std::chrono::high_resolution_clock::now();
    auto norm = Utils::CubicNorm(x - SolveCholesky(LT, D, b));
    end = std::chrono::high_resolution_clock::now();

    auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    return {t1, t2, norm};
}


std::pair<double, double> task6(const Matrix& A, const Vector& b, const Vector& x) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    start = std::chrono::high_resolution_clock::now();
    auto norm = Utils::CubicNorm(x - SolveRelaxation(A, b, EPS, W));
    end = std::chrono::high_resolution_clock::now();

    auto t = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    return {norm, t};
}

std::pair<double, double> task7(Matrix A, Vector b, const Vector &x)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    start = std::chrono::high_resolution_clock::now();
    auto norm = Utils::CubicNorm(x - SolveHouseholder(std::move(A), std::move(b)));
    end = std::chrono::high_resolution_clock::now();

    auto t = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    return {norm, t};
}

std::pair<double, double> task8(Matrix A, Vector b, const Vector &x)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    auto copyA(A);
    auto copyB(b);

    start = std::chrono::high_resolution_clock::now();
    auto norm = Utils::EuclideanNorm(copyA * SolveLeastSquares(std::move(A), std::move(b), true) - copyB);
    end = std::chrono::high_resolution_clock::now();

    auto t = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    return {norm, t};
}

std::pair<double, double> task9(Matrix A, Vector b, const Vector &x)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    start = std::chrono::high_resolution_clock::now();
    auto norm = Utils::CubicNorm(x - SolveGMRES(std::move(A), std::move(b), EPS));
    end = std::chrono::high_resolution_clock::now();

    auto t = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    return {norm, t};
}

std::pair<double, double> task10(Matrix A, Vector b, const Vector &x)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    start = std::chrono::high_resolution_clock::now();
    auto norm = Utils::CubicNorm(x - SolveArnoldiGMRES(std::move(A), std::move(b), EPS));
    end = std::chrono::high_resolution_clock::now();

    auto t = std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count();

    return {norm, t};
}

Matrix cycle(int count = 100, int n = 256, const std::string& out = "reports\\report.txt")
{
	std::ofstream fout(out);
    Matrix maxMatrix(n);

    double minCond, maxCond, avgCond;
    double invTime = 0;
    minCond = std::numeric_limits<double>::max();
    maxCond = std::numeric_limits<double>::min();
    avgCond = 0;

    double minGaussN, maxGaussN, avgGaussN;
    double gaussTime = 0;
    minGaussN = std::numeric_limits<double>::max();
    maxGaussN = std::numeric_limits<double>::min();
    avgGaussN = 0;

    double minLUPN, maxLUPN, avgLUPN;
    double solveTimeLUP = 0;
    double buildTimeLUP = 0;
    minLUPN = std::numeric_limits<double>::max();
    maxLUPN = std::numeric_limits<double>::min();
    avgLUPN = 0;

    double minCholeskyN, maxCholeskyN, avgCholeskyN;
    double solveTimeCholesky = 0;
    double buildTimeCholesky = 0;
    minCholeskyN = std::numeric_limits<double>::max();
    maxCholeskyN = std::numeric_limits<double>::min();
    avgCholeskyN = 0;

    double minRelaxationN, maxRelaxationN, avgRelaxationN;
    double relaxationTime = 0;
    minRelaxationN = std::numeric_limits<double>::max();
    maxRelaxationN = std::numeric_limits<double>::min();
    avgRelaxationN = 0;

    double minHouseHolderN, maxHouseHolderN, avgHouseHolderN;
    double houseHolderTime = 0;
    minHouseHolderN = std::numeric_limits<double>::max();
    maxHouseHolderN = std::numeric_limits<double>::min();
    avgHouseHolderN = 0;

    double minLeastSquaresN, maxLeastSquaresN, avgLeastSquaresN;
    double LeastSquaresTime = 0;
    minLeastSquaresN = std::numeric_limits<double>::max();
    maxLeastSquaresN = std::numeric_limits<double>::min();
    avgLeastSquaresN = 0;

    double minGMRESN, maxGMRESN, avgGMRESN;
    double timeGMRES = 0;
    minGMRESN = std::numeric_limits<double>::max();
    maxGMRESN = std::numeric_limits<double>::min();
    avgGMRESN = 0;

    double minArnoldiGMRESN, maxArnoldiGMRESN, avgArnoldiGMRESN;
    double timeArnoldiGMRES = 0;
    minArnoldiGMRESN = std::numeric_limits<double>::max();
    maxArnoldiGMRESN = std::numeric_limits<double>::min();
    avgArnoldiGMRESN = 0;

    for (int i = 0; i < count; i++)
    {
        Matrix A = Matrix::GenerateMatrix(n, Matrix::variant);
        Vector x = Utils::GenerateVector(n, Matrix::variant);
        Vector b = A * x;

        auto[condition, t2] = task2(A);
        minCond = std::min(condition, minCond);
        if (condition > maxCond)
        {
            maxCond = condition;
            maxMatrix = A;
        }
        avgCond += condition;
        invTime += t2;

        auto[norm3, t3] = task3(A, b, x);
        minGaussN = std::min(norm3, minGaussN);
        maxGaussN = std::max(norm3, maxGaussN);
        avgGaussN += norm3;
        gaussTime += t3;

        auto[t4Build, t4Solve, norm4] = task4(A, b, x);
        minLUPN = std::min(norm4, minLUPN);
        maxLUPN = std::max(norm4, maxLUPN);
        avgLUPN += norm4;
        buildTimeLUP += t4Build;
        solveTimeLUP += t4Solve;

        auto [t5Build, t5Solve, norm5] = task5(A, b, x);
        minCholeskyN = std::min(norm5, minCholeskyN);
        maxCholeskyN = std::max(norm5, maxCholeskyN);
        avgCholeskyN += norm5;
        buildTimeCholesky += t5Build;
        solveTimeCholesky += t5Solve;

        auto[norm6, t6] = task6(A, b, x);
        minRelaxationN = std::min(norm6, minRelaxationN);
        maxRelaxationN = std::max(norm6, maxRelaxationN);
        avgRelaxationN += norm6;
        relaxationTime += t6;

        auto[norm7, t7] = task7(A, b, x);
        minHouseHolderN = std::min(norm7, minHouseHolderN);
        maxHouseHolderN = std::max(norm7, maxHouseHolderN);
        avgHouseHolderN += norm7;
        houseHolderTime += t7;

        auto[norm8, t8] = task8(A.GetSubMatrix(n, 20 * Matrix::variant), b, x);
        minLeastSquaresN = std::min(norm8, minLeastSquaresN);
        maxLeastSquaresN = std::max(norm8, maxLeastSquaresN);
        avgLeastSquaresN += norm8;
        LeastSquaresTime += t8;

        auto[norm9, t9] = task9(A, b, x);
        minGMRESN = std::min(norm9, minGMRESN);
        maxGMRESN = std::max(norm9, maxGMRESN);
        avgGMRESN += norm9;
        timeGMRES += t9;

        auto[norm10, t10] = task10(A, b, x);
        minArnoldiGMRESN = std::min(norm10, minArnoldiGMRESN);
        maxArnoldiGMRESN = std::max(norm10, maxArnoldiGMRESN);
        avgArnoldiGMRESN += norm10;
        timeArnoldiGMRES += t10;

    }

	fout << "T2 - Condition number: " << std::endl;
	fout << "Min number of condition: " << minCond << std::endl;
	fout << "Max number of condition: " << maxCond << std::endl;
	fout << "Avg number of condition: " << avgCond / count << std::endl;
	fout << "Avg time of inverse: " << invTime / count << std::endl;
	fout << std::endl;

	fout << "T3 - Gauss: " << std::endl;
	fout << "Min norm: " << minGaussN << std::endl;
	fout << "Max norm: " << maxGaussN << std::endl;
	fout << "Avg norm: " << avgGaussN / count << std::endl;
	fout << "Avg time of Gauss: " << gaussTime / count << std::endl;
	fout << std::endl;

	fout << "T4 - LUP: " << std::endl;
	fout << "Min norm: " << minLUPN << std::endl;
	fout << "Max norm: " << maxLUPN << std::endl;
	fout << "Avg norm: " << avgLUPN / count << std::endl;
	fout << "Avg time of build LUP: " << buildTimeLUP / count << std::endl;
	fout << "Avg time of solve LUP: " << solveTimeLUP / count << std::endl;
	fout << std::endl;

	fout << "T5 - Cholesky: " << std::endl;
	fout << "Min norm: " << minCholeskyN << std::endl;
	fout << "Max norm: " << maxCholeskyN << std::endl;
	fout << "Avg norm: " << avgCholeskyN / count << std::endl;
	fout << "Avg time of build Cholesky: " << buildTimeCholesky / count << std::endl;
	fout << "Avg time of solve Cholesky: " << solveTimeCholesky / count << std::endl;
	fout << std::endl;

	fout << "T6 - Relaxation: " << std::endl;
	fout << "Min norm: " << minRelaxationN << std::endl;
	fout << "Max norm: " << maxRelaxationN << std::endl;
	fout << "Avg norm: " << avgRelaxationN / count << std::endl;
	fout << "Avg time of Relaxation: " << relaxationTime / count << std::endl;
	fout << std::endl;

	fout << "T7 - Householder: " << std::endl;
	fout << "Min norm: " << minHouseHolderN << std::endl;
	fout << "Max norm: " << maxHouseHolderN << std::endl;
	fout << "Avg norm: " << avgHouseHolderN / count << std::endl;
	fout << "Avg time of Householder: " << houseHolderTime / count << std::endl;
	fout << std::endl;

	fout << "T8 - LeastSquares (QR): " << std::endl;
	fout << "Min norm: " << minLeastSquaresN << std::endl;
	fout << "Max norm: " << maxLeastSquaresN << std::endl;
	fout << "Avg norm: " << avgLeastSquaresN / count << std::endl;
	fout << "Avg time of LeastSquares: " << LeastSquaresTime / count << std::endl;
	fout << std::endl;

	fout << "T9 - GMRES: " << std::endl;
	fout << "Min norm: " << minGMRESN << std::endl;
	fout << "Max norm: " << maxGMRESN << std::endl;
	fout << "Avg norm: " << avgGMRESN / count << std::endl;
	fout << "Avg time of GMRES: " << timeGMRES / count << std::endl;
	fout << std::endl;

	fout << "T10 - GMRES (Arnoldi): " << std::endl;
	fout << "Min norm: " << minArnoldiGMRESN << std::endl;
	fout << "Max norm: " << maxArnoldiGMRESN << std::endl;
	fout << "Avg norm: " << avgArnoldiGMRESN / count << std::endl;
	fout << "Avg time of GMRES (Arnoldi): " << timeArnoldiGMRES / count << std::endl;
	fout << std::endl;

    return maxMatrix;
}



void tests(const Matrix& A, const Vector& b, const Vector& x)
{
    std::cout << "Condition number:" << std::endl;
    std::cout << GetConditionNumber(A);
    std::cout << std::endl;

    std::cout << "Gauss: " << std::endl;
    std::cout << SolveGauss(A, b);
    std::cout << std::endl;

    std::cout << "LUP: " << std::endl;
    auto [LU, P] = BuildLUP(A);
    std::cout << SolveLUP(LU, P, b);
    std::cout << std::endl;

    std::cout << "Cholesky:" << std::endl;
    auto [LT, D] = BuildCholesky(A);
    std::cout <<  SolveCholesky(LT, D, b);
    std::cout << std::endl;

    std::cout << "Relaxation:" << std::endl;
    std::cout << SolveRelaxation(A, b, EPS, W);
    std::cout << std::endl;

    std::cout << "Householder:" << std::endl;
    std::cout << SolveHouseholder(A, b);
    std::cout << std::endl;

    std::cout << "LeastSquares (QR):" << std::endl;
    std::cout << SolveLeastSquares(A, b, true);
    std::cout << std::endl;

    std::cout << "GMRES:" << std::endl;
    std::cout << SolveGMRES(A, b, EPS);
    std::cout << std::endl;

	std::cout << "GMRES (Arnoldi)" << std::endl;
    std::cout <<  SolveArnoldiGMRES(A, b, EPS);
	std::cout << std::endl;


}


int main()
{
    int n = 256;
    double arr[] = {
            7, 1, 2, 3,
            1, 15, 3, 6,
            2, 3, 16, 8,
            3, 6, 8, 20,
    };
	Matrix A = Matrix::FromArray(arr, 4, 4);
    Vector x = {1, 2, 3, 4};
    Vector b = A * x;

	tests(A, b, x);
    return 0;
}
