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

#ifndef LUT_OPS_KERNELS_MATMUL_LUT_OP_KERNEL_H_
#define LUT_OPS_KERNELS_MATMUL_LUT_OP_KERNEL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/hash/hash.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace tensorflow {
namespace functor {

// Helpers to define tensor<T> needed by MatMulLUT op.
template <typename T>
struct MatMulLUTTypes {
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>
      out_type;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                           Eigen::Aligned>
      in_type;
};

template <typename Device, typename In0, typename In1, typename Lut, 
          typename Out, typename DimPair>
void MatMulLUT(const Device& d, Out out, In0 in0, In1 in1, Lut lut,
            const DimPair& dim_pair) {
  out.device(d) = in0.contract(in1, dim_pair);
}

template <typename Device, typename T, typename U>
struct MatMulLUTFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, typename MatMulLUTTypes<U>::out_type out,
      typename MatMulLUTTypes<T>::in_type in0,
      typename MatMulLUTTypes<T>::in_type in1,
      typename MatMulLUTTypes<U>::in_type lut,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair);
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // LUT_OPS_KERNELS_MATMUL_LUT_OP_KERNEL_H_
