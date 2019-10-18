/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Note this header is used in both TF and TFLite.
namespace Eigen {


// these are the same as tensorflow::TTypes
template<typename T, int n_dims = 1>
using MyTensor = Tensor<T, n_dims, RowMajor, DenseIndex>;
template<typename T, int n_dims = 1>
using MyTensorMap = TensorMap<MyTensor<T, n_dims>, Aligned>;

template <typename Input, typename Kernel, typename LookupTable,
          typename OutputKernel = const NoOpOutputKernel>
EIGEN_DEVICE_FUNC
    EIGEN_ALWAYS_INLINE const MyTensorMap<typename std::remove_const<typename internal::traits<LookupTable>::Scalar>::type, 4>
    SpatialConvolutionLUT(const Input& input, const Kernel& kernel, const LookupTable& lookupTable,
                       const Index row_stride = 1, const Index col_stride = 1,
                       const PaddingType padding_type = PADDING_SAME,
                       const Index row_in_stride = 1,
                       const Index col_in_stride = 1,
                       const OutputKernel& output_kernel = OutputKernel()) {

  using LutValue = typename std::remove_const<typename internal::traits<LookupTable>::Scalar>::type;
  using LutIndex = typename std::remove_const<typename internal::traits<Input>::Scalar>::type;
  using TensorIndex = typename internal::traits<Input>::Index;

  TensorRef<Tensor<typename internal::traits<Input>::Scalar,
                   internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);
  TensorRef<Tensor<typename internal::traits<Kernel>::Scalar,
                   internal::traits<Kernel>::NumDimensions,
                   internal::traits<Kernel>::Layout, TensorIndex> >
      kern(kernel);

  EIGEN_STATIC_ASSERT(
      internal::traits<Input>::Layout == internal::traits<Kernel>::Layout,
      YOU_MADE_A_PROGRAMMING_MISTAKE)
  const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  const int NumDims = internal::traits<Input>::NumDimensions;

  // Number of filters to apply. This is the same as the output depth of the
  // result
  const TensorIndex kernelFilters =
      isColMajor ? kern.dimensions()[0] : kern.dimensions()[3];
  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels =
      isColMajor ? kern.dimensions()[1] : kern.dimensions()[2];
  const TensorIndex kernelRows =
      isColMajor ? kern.dimensions()[2] : kern.dimensions()[1];
  const TensorIndex kernelCols =
      isColMajor ? kern.dimensions()[3] : kern.dimensions()[0];

  const Index kernelRowsEff =
      kernelRows + (kernelRows - 1) * (row_in_stride - 1);
  const Index kernelColsEff =
      kernelCols + (kernelCols - 1) * (col_in_stride - 1);

  array<IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = IndexPair<TensorIndex>(1, 0);

  const TensorIndex InputRows =
      isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex InputCols =
      isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);

  TensorIndex out_height {0};
  TensorIndex out_width {0};
  switch (padding_type) {
    case PADDING_VALID: {
      out_height = numext::ceil((InputRows - kernelRowsEff + 1.f) /
                                static_cast<float>(row_stride));
      out_width = numext::ceil((InputCols - kernelColsEff + 1.f) /
                               static_cast<float>(col_stride));
      break;
    }
    case PADDING_SAME: {
      out_height = numext::ceil(InputRows / static_cast<float>(row_stride));
      out_width = numext::ceil(InputCols / static_cast<float>(col_stride));
      break;
    }
    default: {
      eigen_assert(false && "unexpected padding");
    }
  }

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  // kernels
  // - the second dimension (dims[1]): everything else
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[1] = out_height * out_width;
    for (int i = 3; i < NumDims; ++i) {
      pre_contract_dims[1] *= in.dimension(i);
    }
  } else {
    pre_contract_dims[1] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[0] = out_height * out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      pre_contract_dims[0] *= in.dimension(i);
    }
  }

  // Molds the output of the contraction into the shape expected by the used
  // (assuming this is ColMajor):
  // - 1st dim: kernel filters
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelFilters;
    post_contract_dims[1] = out_height;
    post_contract_dims[2] = out_width;
    for (int i = 3; i < NumDims; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  } else {
    post_contract_dims[NumDims - 1] = kernelFilters;
    post_contract_dims[NumDims - 2] = out_height;
    post_contract_dims[NumDims - 3] = out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  }

  DSizes<TensorIndex, 2> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels * kernelRows * kernelCols;
  } else {
    kernel_dims[0] = kernelChannels * kernelRows * kernelCols;
    kernel_dims[1] = kernelFilters;
  }

  int n_patches = isColMajor ? pre_contract_dims[1] : pre_contract_dims[0];
  int patch_size = isColMajor ? pre_contract_dims[0] : pre_contract_dims[1];

  const int padValueRowIdx = 
    isColMajor ? (lookupTable.dimensions()[1] - 1) / 2 : (lookupTable.dimensions()[0] - 1) / 2;

  MyTensor<LutIndex, 2> kernel_reshaped = kernel.reshape(kernel_dims);
  MyTensor<LutIndex, 2> input_patches_reshaped = input.extract_image_patches(
      kernelRows, kernelCols, row_stride, col_stride,
      row_in_stride, col_in_stride, padding_type, padValueRowIdx)
    .reshape(pre_contract_dims);

  MyTensor<LutValue, 4> result (post_contract_dims);
  /* MyTensorMap<LutValue, 2> resultMap2D (result.data(), kernelFilters, n_patches); */
  MyTensorMap<LutValue, 2> resultMap2D (result.data(), n_patches, kernelFilters);

  for (int i {0}; i < kernelFilters; ++i)
  {
    for (int j {0}; j < n_patches; ++j)
    {
      LutValue tmp {0};
      for (int k {0}; k < patch_size; ++k)
      {
        tmp += lookupTable(input_patches_reshaped(j, k), kernel_reshaped(k, i));
      }
      resultMap2D(j, i) = tmp;
    }
  }
  return result;
}

}  // end namespace Eigen
