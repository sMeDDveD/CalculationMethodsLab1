cmake_minimum_required(VERSION 3.15)
project(CalculationMethods)

set(CMAKE_CXX_STANDARD 20)

include_directories(Practice)

add_executable(CalculationMethods
        Practice/Cholesky.cpp
        Practice/Cholesky.h
        Practice/ConditionNumber.cpp
        Practice/ConditionNumber.h
        Practice/Gauss.cpp
        Practice/Gauss.h
        Practice/Householder.cpp
        Practice/Householder.h
        Practice/LeastSquares.cpp
        Practice/LeastSquares.h
        Practice/LUP.cpp
        Practice/LUP.h
        Practice/Main.cpp
        Practice/Matrix.cpp
        Practice/Matrix.h
        Practice/Relaxation.cpp
        Practice/Relaxation.h
        Practice/Utils.cpp
        Practice/Utils.h
        Practice/GMRES.h
        Practice/GMRES.cpp
        Practice/ArnoldiGMRES.h
        Practice/ArnoldiGMRES.cpp)
