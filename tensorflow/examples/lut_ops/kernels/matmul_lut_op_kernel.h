/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

#include "tensorflow/examples/lut_ops/kernels/eigen_matmul_lut-inl.h"

namespace lut_ops {
namespace functor {

using namespace tensorflow;

// Helpers to define tensor<T> needed by MatMulLUT op.
template <typename T>
struct MTypes {

  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, 
          Eigen::Aligned>
            Matrix;

  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
          Eigen::Aligned>
            ConstMatrix;
};

template <typename Device, typename In0, typename In1, typename Lut, 
          typename Out, typename DimPair>
void MatMulLUT(const Device& d, Out out, In0 in0, In1 in1, Lut lut,
            const DimPair& dim_pair) {
  out.device(d) = Eigen::MatMulLut(in0, in1, lut, dim_pair);
}

template <typename Device, typename T, typename U>
struct MatMulLUTFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication using lookup table "lut"
  void operator()(
      const Device& d, typename MTypes<U>::Matrix out,
      typename MTypes<T>::ConstMatrix in0,
      typename MTypes<T>::ConstMatrix in1,
      typename MTypes<U>::ConstMatrix lut,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair);
};

} // functor
} // lut_ops
