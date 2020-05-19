cmake_minimum_required(VERSION 3.15)
project(CalculationMethods)

set(CMAKE_CXX_STANDARD 20)

include_directories(Sources)

add_executable(CalculationMethods
        Sources/Cholesky.cpp
        Sources/Cholesky.h
        Sources/ConditionNumber.cpp
        Sources/ConditionNumber.h
        Sources/Gauss.cpp
        Sources/Gauss.h
        Sources/Householder.cpp
        Sources/Householder.h
        Sources/LeastSquares.cpp
        Sources/LeastSquares.h
        Sources/LUP.cpp
        Sources/LUP.h
        Sources/Main.cpp
        Sources/Matrix.cpp
        Sources/Matrix.h
        Sources/Relaxation.cpp
        Sources/Relaxation.h
        Sources/Utils.cpp
        Sources/Utils.h
        Sources/GMRES.h
        Sources/GMRES.cpp
        Sources/ArnoldiGMRES.h
        Sources/ArnoldiGMRES.cpp)
