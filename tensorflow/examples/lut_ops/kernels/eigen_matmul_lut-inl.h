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
  //
// these are the same as tensorflow::TTypes
template<typename T>
using MyMatrix = Tensor<T, 2, RowMajor, DenseIndex>;
template<typename T>
using MyMatrixMap = TensorMap<MyMatrix<T>, Aligned>;

template <typename RowIndices, typename ColIndices, typename LookupTable, typename DimPair>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE 
const MyMatrixMap<typename std::remove_const<typename internal::traits<LookupTable>::Scalar>::type>
  MatMulLut(const RowIndices& row_indices, const ColIndices& col_indices, const LookupTable& lookupTable,
          const DimPair& dim_pair) {

  using LutValue = typename std::remove_const<typename internal::traits<LookupTable>::Scalar>::type;
  using InputIndex = typename internal::traits<RowIndices>::Index;

  TensorRef<Tensor<typename internal::traits<RowIndices>::Scalar,
                   internal::traits<RowIndices>::NumDimensions,
                   internal::traits<RowIndices>::Layout, InputIndex> >
      row_idcs(row_indices);
  TensorRef<Tensor<typename internal::traits<ColIndices>::Scalar,
                   internal::traits<ColIndices>::NumDimensions,
                   internal::traits<ColIndices>::Layout, InputIndex> >
      col_idcs(col_indices);

  EIGEN_STATIC_ASSERT(
      internal::traits<RowIndices>::Layout == internal::traits<ColIndices>::Layout,
      YOU_MADE_A_PROGRAMMING_MISTAKE)

  const InputIndex a_dim_remaining = 1 - dim_pair[0].first;
  const InputIndex b_dim_remaining = 1 - dim_pair[0].second;
  const InputIndex dim_product = row_idcs.dimension(dim_pair[0].first);

  DSizes<InputIndex, 2> output_dims;
  output_dims[0] = row_idcs.dimension(a_dim_remaining);
  output_dims[1] = col_idcs.dimension(b_dim_remaining);

  /* std::cout << "[MatMulLutOp] ouput_dims: (" << output_dims[0] << ", " << output_dims[1] << "), dim_product: " << dim_product << "\n"; */

  MyMatrix<LutValue> result (output_dims);

  using RowIndex = typename internal::traits<RowIndices>::Scalar;
  using ColIndex = typename internal::traits<ColIndices>::Scalar;

  const LutValue* lut = lookupTable.data();
  const RowIndex* rowIdcs = row_indices.data();
  const ColIndex* colIdcs = col_indices.data();

  const InputIndex lutRows = lookupTable.dimension(0);
  const InputIndex lutCols = lookupTable.dimension(1);
  
  Tensor<LutValue, 1, RowMajor, DenseIndex> tmp (dim_product);
  Tensor<LutValue, 0, RowMajor, DenseIndex> aggregated;
  LutValue* tmpData = tmp.data();

  for (InputIndex i {0}; i < output_dims[0]; ++i)
  {
    for (InputIndex j {0}; j < output_dims[1]; ++j)
    {
      for (InputIndex k {0}; k < dim_product; ++k)
      {
        tmpData[k] = lut[rowIdcs[i * dim_product + k] * lutCols + colIdcs[k * output_dims[1] + j]];
      }
      aggregated = tmp.sum();
      result(i, j) = aggregated();
    }
  }
  return result;
}

//-----------------------------------------------------------------------------
template <typename RowIndices, typename ColIndices, typename LookupTable, typename DimPair>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE 
const MyMatrixMap<typename std::remove_const<typename internal::traits<LookupTable>::Scalar>::type>
  MatMulLut_V2(const RowIndices& row_indices, const ColIndices& col_indices, const LookupTable& lookupTable,
          const DimPair& dim_pair) {

  using LutValue = typename std::remove_const<typename internal::traits<LookupTable>::Scalar>::type;
  using InputIndex = typename internal::traits<RowIndices>::Index;
  using InputValueR = typename internal::traits<RowIndices>::Scalar;
  using InputValueC = typename internal::traits<ColIndices>::Scalar;

  TensorRef<Tensor<typename internal::traits<RowIndices>::Scalar,
                   internal::traits<RowIndices>::NumDimensions,
                   internal::traits<RowIndices>::Layout, InputIndex> >
      row_idcs(row_indices);
  TensorRef<Tensor<typename internal::traits<ColIndices>::Scalar,
                   internal::traits<ColIndices>::NumDimensions,
                   internal::traits<ColIndices>::Layout, InputIndex> >
      col_idcs(col_indices);

  EIGEN_STATIC_ASSERT(
      internal::traits<RowIndices>::Layout == internal::traits<ColIndices>::Layout,
      YOU_MADE_A_PROGRAMMING_MISTAKE)

  const InputIndex a_dim_remaining = 1 - dim_pair[0].first;
  const InputIndex b_dim_remaining = 1 - dim_pair[0].second;
  const InputIndex dim_product = row_idcs.dimension(dim_pair[0].first);

  DSizes<InputIndex, 2> output_dims;
  output_dims[0] = row_idcs.dimension(a_dim_remaining);
  output_dims[1] = col_idcs.dimension(b_dim_remaining);

  // swap layout of col_indices matrix to ColMajor to access items linearly
  array<int, 2> shuffle {{1, 0}};
  Tensor<InputValueC, 2, ColMajor, DenseIndex> colIdcsCM = col_indices.swap_layout().shuffle(shuffle);

  const InputValueR n_cols_lut = lookupTable.dimension(1);
  MyMatrix<InputValueR> rowIdcs_ = row_indices * n_cols_lut;

  // generate a vector ks = [0, 1, 2, ..., dim_product-1]
  // (the indices of the innermost loop)
  const int scan_axis {0};
  Tensor<InputIndex, 1, RowMajor, DenseIndex> ks (dim_product);
  ks = ks.setConstant(1).cumsum(scan_axis) - InputIndex(1);

  Tensor<LutValue, 1, RowMajor, DenseIndex> tmp (dim_product);
  Tensor<LutValue, 0, RowMajor, DenseIndex> aggregated;
  MyMatrix<LutValue> result (output_dims);

  const LutValue* lut = lookupTable.data();
  LutValue* tmpData = tmp.data();
  const InputValueR* rowIdcs = rowIdcs_.data();
  const InputValueC* colIdcs = colIdcsCM.data();

  Tensor<InputIndex, 1, RowMajor, DenseIndex> rii (dim_product);
  Tensor<InputIndex, 1, RowMajor, DenseIndex> cii (dim_product);

  for (InputIndex i {0}; i < output_dims[0]; ++i)
  {
    rii = ks + (i * dim_product);
    for (InputIndex j {0}; j < output_dims[1]; ++j)
    {
      cii = ks + (j * dim_product);
      for (InputIndex k {0}; k < dim_product; ++k)
      {
        // compute lut[row_indices[i * dim_product + k] * lutCols + col_indices[k * output_dims[1] + j]];
        tmpData[k] = lut[rowIdcs[rii[k]] + colIdcs[cii[k]]];
      }
      aggregated = tmp.sum();
      result(i, j) = aggregated();
    }
  }
  return result;
}

}  // end namespace Eigen
