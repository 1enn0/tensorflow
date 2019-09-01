/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef LUT_OPS_KERNELS_CONV2D_LUT_OP_KERNEL_H_
#define LUT_OPS_KERNELS_CONV2D_LUT_OP_KERNEL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/tensor_format.h"

using namespace tensorflow;

template <typename Device, typename T, typename U>
struct LaunchConv2DLUTOp {
  void operator()(OpKernelContext* ctx,
                  const Tensor& input, const Tensor& filter, const Tensor& lut,
                  int row_dilation, int col_dilation, 
                  int row_stride, int col_stride,
                  const Padding& padding,
                  Tensor* output,
                  TensorFormat data_format);
};
//
// Convolution dimensions inferred from parameters, input and filter tensors.
struct Conv2DLUTDimensions {
  int batch;
  int input_rows;
  int input_cols;
  int in_depth;

  int filter_rows;
  int filter_cols;
  int patch_depth;
  int out_depth;

  int stride_rows;
  int stride_cols;

  int dilation_rows;
  int dilation_cols;

  int64 out_rows;
  int64 out_cols;
  int64 pad_rows_before;
  int64 pad_rows_after;
  int64 pad_cols_before;
  int64 pad_cols_after;
};
//
// Convolution parameters specified by Op attributes.
struct Conv2DLUTParameters {
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format;
};


// Initializes and validates Conv2DLUT parameters configured by OpKernel
// attributes.
Status InitConv2DLUTParameters(const OpKernelConstruction* context,
                            Conv2DLUTParameters* params);

// Computes and validates convolutions dimensions from Conv2DLUT parameters. If
// parameters are valid, dimensions will be updated with derived convolution
// dimensions, otherwise an error will be returned.
Status ComputeConv2DLUTDimension(const Conv2DLUTParameters& params,
                              const Tensor& input, const Tensor& filter,
                              Conv2DLUTDimensions* dimensions);

#endif  // LUT_OPS_KERNELS_CONV2D_LUT_OP_KERNEL_H_
