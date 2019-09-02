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
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/examples/lut_ops/kernels/eigen_spatial_convolutions_lut-inl.h"
#include "tensorflow/core/util/tensor_format.h"

namespace lut_ops {
namespace functor {

using namespace tensorflow;

template <typename Device, typename T, typename U,
          typename OutputKernel = const Eigen::NoOpOutputKernel>
struct SpatialConvolutionLUT {
  using ConstTensorT4 = typename TTypes<T, 4>::ConstTensor;
  using ConstTensorU2 = typename TTypes<U, 2>::ConstTensor;
  using TensorU4 = typename TTypes<U, 4>::Tensor;
  void operator()(const Device& d, 
                  TensorU4 output,
                  ConstTensorT4 input,
                  ConstTensorT4 filter, 
                  ConstTensorU2 lut,
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

} // functor
} // lut_ops
