#pragma once

#include "Eigen/Eigen"
#include "unsupported/Eigen/CXX11/Tensor"

using int32 = long int;

template <typename T>
using Mat = Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>;

using MatI = Mat<int32>;
using MatF = Mat<float>;

template <typename T>
using Vec = Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>;

using VecI = Vec<int32>;
using VecF = Vec<float>;
