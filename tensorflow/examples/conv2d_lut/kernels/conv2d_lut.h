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

#ifndef CONV2D_LUT_KERNELS_CONV_2D_H_
#define CONV2D_LUT_KERNELS_CONV_2D_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/examples/conv2d_lut/kernels/eigen_spatial_convolutions_lut-inl.h"
#include "tensorflow/core/util/tensor_format.h"

namespace functor {

using namespace tensorflow;

template <typename Device, typename T, typename U,
          typename OutputKernel = const Eigen::NoOpOutputKernel>
struct SpatialConvolutionLUT {
  void operator()(const Device& d, typename TTypes<U, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, 
                  typename TTypes<U, 2>::ConstTensor lut,
                  int row_stride, int col_stride, int row_dilation, int col_dilation,
                  const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    // Need to swap row/col when calling Eigen. Eigen expects the tensor
    // in NWHC format, but the tensor given is in NHWC.
    output.device(d) = Eigen::SpatialConvolutionLUT(
        input, filter, lut, col_stride, row_stride,
        padding, col_dilation, row_dilation, output_kernel);
  }
};

// TODO(vrv): Figure out how to use the MatMulFunctor in matmul_op.h.
// My initial attempt to do this compiled but failed in the pytest
// due to a swigdeps error.
template <typename Device, typename T,
          typename OutputKernel = const Eigen::NoOpOutputKernel>
struct MatMulConvFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, typename TTypes<T, 2>::Tensor out,
      typename TTypes<T, 2>::ConstTensor in0,
      typename TTypes<T, 2>::ConstTensor in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      const OutputKernel& output_kernel = OutputKernel()) {
    out.device(d) = in0.contract(in1, dim_pair, output_kernel);
  }
};

}  // namespace functor

#endif  // CONV2D_LUT_KERNELS_CONV_2D_H_
