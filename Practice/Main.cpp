#include <iostream>
#include <functional>
#include <chrono>
#include <utility>

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

constexpr double EPS = 0.000001;
constexpr double W = 10.0 / 6;

double timeit(Matrix A, Vector b, Vector x) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    start = std::chrono::high_resolution_clock::now();
    std::cout << Utils::EuclideanNorm(x - SolveGauss(std::move(A), std::move(b)))
    << std::endl;
    end = std::chrono::high_resolution_clock::now();
    return 0;
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

    auto copyA = A;
    auto copyB = b;

    start = std::chrono::high_resolution_clock::now();
    auto norm = Utils::CubicNorm(copyA * SolveLeastSquares(std::move(A), std::move(b), true) - copyB);
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

void cycle(int count = 100, int n = 256)
{
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
        Vector x(n);
        std::iota(x.begin(), x.end(), 1);
        Vector b = A * x;

        auto[condition, t2] = task2(A);
        minCond = std::min(condition, minCond);
        maxCond = std::max(condition, maxCond);
        avgCond += condition;
        invTime += t2;

        auto [norm3, t3] = task3(A, b, x);
        minGaussN = std::min(norm3, minGaussN);
        maxGaussN = std::max(norm3, maxGaussN);
        avgGaussN += norm3;
        gaussTime += t3;

        auto [t4Build, t4Solve, norm4] = task4(A, b, x);
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

        auto[norm8, t8] = task8(A.GetSubMatrix(n, n / 2), b, x);
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

    std::cout << "T2 - Condition number: " << std::endl;
    std::cout << "Min number of condition: " << minCond << std::endl;
    std::cout << "Max number of condition: " << maxCond << std::endl;
    std::cout << "Avg number of condition: " << avgCond / count << std::endl;
    std::cout << "Avg time of inverse: " << invTime / count << std::endl;
    std::cout << std::endl;

    std::cout << "T3 - Gauss: " << std::endl;
    std::cout << "Min norm: " << minGaussN << std::endl;
    std::cout << "Max norm: " << maxGaussN << std::endl;
    std::cout << "Avg norm: " << avgGaussN / count << std::endl;
    std::cout << "Avg time of Gauss: " << gaussTime / count << std::endl;
    std::cout << std::endl;

    std::cout << "T4 - LUP: " << std::endl;
    std::cout << "Min norm: " << minLUPN << std::endl;
    std::cout << "Max norm: " << maxLUPN << std::endl;
    std::cout << "Avg norm: " << avgLUPN / count << std::endl;
    std::cout << "Avg time of build LUP: " << buildTimeLUP / count << std::endl;
    std::cout << "Avg time of solve LUP: " << solveTimeLUP / count << std::endl;
    std::cout << std::endl;

    std::cout << "T5 - Cholesky: " << std::endl;
    std::cout << "Min norm: " << minCholeskyN << std::endl;
    std::cout << "Max norm: " << maxCholeskyN << std::endl;
    std::cout << "Avg norm: " << avgCholeskyN / count << std::endl;
    std::cout << "Avg time of build Cholesky: " << buildTimeCholesky / count << std::endl;
    std::cout << "Avg time of solve Cholesky: " << solveTimeCholesky / count << std::endl;
    std::cout << std::endl;

    std::cout << "T6 - Relaxation: " << std::endl;
    std::cout << "Min norm: " << minRelaxationN << std::endl;
    std::cout << "Max norm: " << maxRelaxationN << std::endl;
    std::cout << "Avg norm: " << avgRelaxationN / count << std::endl;
    std::cout << "Avg time of Relaxation: " << relaxationTime / count << std::endl;
    std::cout << std::endl;

    std::cout << "T7 - Householder: " << std::endl;
    std::cout << "Min norm: " << minHouseHolderN << std::endl;
    std::cout << "Max norm: " << maxHouseHolderN << std::endl;
    std::cout << "Avg norm: " << avgHouseHolderN / count << std::endl;
    std::cout << "Avg time of Householder: " << houseHolderTime / count << std::endl;
    std::cout << std::endl;

    std::cout << "T8 - LeastSquares (QR): " << std::endl;
    std::cout << "Min norm: " << minLeastSquaresN << std::endl;
    std::cout << "Max norm: " << maxLeastSquaresN << std::endl;
    std::cout << "Avg norm: " << avgLeastSquaresN / count << std::endl;
    std::cout << "Avg time of LeastSquares: " << LeastSquaresTime / count << std::endl;
    std::cout << std::endl;

    std::cout << "T9 - GMRES: " << std::endl;
    std::cout << "Min norm: " << minGMRESN << std::endl;
    std::cout << "Max norm: " << maxGMRESN << std::endl;
    std::cout << "Avg norm: " << avgGMRESN / count << std::endl;
    std::cout << "Avg time of GMRES: " << timeGMRES / count << std::endl;
    std::cout << std::endl;

    std::cout << "T10 - GMRES (Arnoldi): " << std::endl;
    std::cout << "Min norm: " << minArnoldiGMRESN << std::endl;
    std::cout << "Max norm: " << maxArnoldiGMRESN << std::endl;
    std::cout << "Avg norm: " << avgArnoldiGMRESN / count << std::endl;
    std::cout << "Avg time of GMRES (Arnoldi): " << timeArnoldiGMRES / count << std::endl;
    std::cout << std::endl;


}

void tests(const Matrix& A, const Vector& b, const Vector& x)
{
    std::cout << "Condition number:" << std::endl;
    std::cout << GetConditionNumber(A);
    std::cout << std::endl;

    std::cout << "Gauss: " << std::endl;
    std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveGauss(A, b)));
    std::cout << std::endl;

    std::cout << "LUP: " << std::endl;
    auto [LU, P] = BuildLUP(A);
    std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveLUP(LU, P, b)));
    std::cout << std::endl;

    std::cout << "Cholesky:" << std::endl;
    auto [LT, D] = BuildCholesky(A);
    std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveCholesky(LT, D, b)));
    std::cout << std::endl;

    std::cout << "Relaxation:" << std::endl;
    std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveRelaxation(A, b, EPS, 1.2)));
    std::cout << std::endl;

    std::cout << "Householder:" << std::endl;
    std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveHouseholder(A, b)));
    std::cout << std::endl;

    std::cout << "LeastSquares (GaussT):" << std::endl;
    std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveLeastSquares(A, b, false)));
    std::cout << std::endl;

    std::cout << "LeastSquares (QR):" << std::endl;
    std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveLeastSquares(A, b, true)));
    std::cout << std::endl;

    std::cout << "GMRES:" << std::endl;
    std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveGMRES(A, b, EPS)));
    std::cout << std::endl;

	std::cout << "GMRES (Arnoldi)" << std::endl;
    std::cout << Utils::EuclideanNorm(Utils::SubVectors(x, SolveArnoldiGMRES(A, b, EPS)));
	std::cout << std::endl;


}


int main()
{
    int n = 256;
    double arr[] = {
            5, 10, 3,
            -10, 15, -4,
            5, 8, 16
    };
    Matrix A = Matrix::GenerateMatrix(n, Matrix::variant);
    Vector x(A.GetCols());
    std::iota(x.begin(), x.end(), 1);
    Vector b = A * x;

    Matrix testMatrix = Matrix::FromArray(arr, 3, 3);

    cycle(10, 100);
    //tests(A, b, x);
    system("pause");
    return 0;
}
