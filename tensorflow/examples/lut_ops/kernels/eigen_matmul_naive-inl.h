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

template <typename InA, typename InB, typename DimPair>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE 
const MyMatrixMap<typename std::remove_const<typename internal::traits<InA>::Scalar>::type>
  MatMulNaive(const InA& a, const InB& b, const DimPair& dim_pair) {

  using InputIndex = typename internal::traits<InA>::Index;
  using InputValue = typename internal::traits<InA>::Scalar;
  using OutputValue = typename std::remove_const<typename internal::traits<InA>::Scalar>::type;

  TensorRef<Tensor<typename internal::traits<InA>::Scalar,
                   internal::traits<InA>::NumDimensions,
                   internal::traits<InA>::Layout, InputIndex> >
      in_a(a);
  TensorRef<Tensor<typename internal::traits<InB>::Scalar,
                   internal::traits<InB>::NumDimensions,
                   internal::traits<InB>::Layout, InputIndex> >
      in_b(b);

  /* EIGEN_STATIC_ASSERT( */
  /*     internal::traits<InA>::Layout == internal::traits<InB>::Layout, */
  /*     YOU_MADE_A_PROGRAMMING_MISTAKE) */

  const bool isARowMajor = internal::traits<InA>::Layout == Eigen::RowMajor;
  const bool isBRowMajor = internal::traits<InB>::Layout == Eigen::RowMajor;

  const InputIndex a_dim_remaining = 1 - dim_pair[0].first;
  const InputIndex b_dim_remaining = 1 - dim_pair[0].second;
  const InputIndex dim_product = in_a.dimension(dim_pair[0].first);

  DSizes<InputIndex, 2> output_dims;
  output_dims[0] = in_a.dimension(a_dim_remaining);
  output_dims[1] = in_b.dimension(b_dim_remaining);

  /* std::cout << "[MatMulNaiveOp] ouput_dims: (" << output_dims[0] << ", " << output_dims[1] << "), dim_product: " << dim_product << "\n"; */

  const InputValue* pA = a.data();
  const InputValue* pB = b.data();
  Tensor<OutputValue, 1, RowMajor, DenseIndex> tmp (dim_product);
  Tensor<OutputValue, 0, RowMajor, DenseIndex> aggregated;
  OutputValue* tmpData = tmp.data();

  MyMatrix<OutputValue> result (output_dims);
  for (InputIndex i {0}; i < output_dims[0]; ++i)
  {
    for (InputIndex j {0}; j < output_dims[1]; ++j)
    {
      for (InputIndex k {0}; k < dim_product; ++k)
      {
        const size_t a_idx = isARowMajor ? i * dim_product + k : k * output_dims[0] + i;
        const size_t b_idx = isARowMajor ? k * output_dims[1] + j : j * dim_product + k;
        /* tmp += a(i, k) * b(k, j); */
        /* tmpData[k] = pA[i * dim_product + k] * pB[k * output_dims[1] + j]; */
        tmpData[k] = pA[a_idx] * pB[b_idx];
      }
      aggregated = tmp.sum();
      result(i, j) = aggregated();
    }
  }
  return result;
}

}  // end namespace Eigen
