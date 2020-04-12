#pragma once

#include <utility>

#include "Matrix.h"
#include "Utils.h"


std::pair<Matrix, Vector> BuildCholesky(Matrix m);

Vector SolveCholesky(Matrix LT, Vector D, Vector b);
