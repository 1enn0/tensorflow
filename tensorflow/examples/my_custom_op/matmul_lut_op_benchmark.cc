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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

template <typename IndexType, typename LUTValueType>
static Graph* MatMulLUT(int m, int k, int n, bool transpose_a, bool transpose_b,
                        DataType indexType,
                        int n_activation, int n_weights, DataType lutValueType) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(indexType, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<IndexType>().setRandom();
  Tensor in1(indexType, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<IndexType>().setRandom();
  Tensor lut(lutValueType, TensorShape({n_activation, n_weights}));
  lut.flat<LUTValueType>().setRandom();
  
  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("mat_mul_lut_op"), "MatMulLUT")
                  .Input(test::graph::Constant(g, in0))
                  .Input(test::graph::Constant(g, in1))
                  .Input(test::graph::Constant(g, lut))
                  .Attr("transpose_a", transpose_a)
                  .Attr("transpose_b", transpose_b)
                  .Finalize(g, &node));
  return g;
}

#define BM_MatMulLUTDev(M, K, N, TA, TB, IDX_TYPE, IDX_TFTYPE, NA, NW, LUT_VAL_TYPE, LUT_VAL_TFTYPE, DEVICE)                       \
  static void BM_MatMulLUT##_##M##_##K##_##N##_##TA##_##TB##_##IDX_TYPE##_##IDX_TFTYPE##_##NA##_##NW##__##LUT_VAL_TYPE##_##LUT_VAL_TFTYPE##_##DEVICE( \
      int iters) {                                                             \
    testing::UseRealTime();                                                    \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);        \
    test::Benchmark(#DEVICE, MatMulLUT<IDX_TYPE, LUT_VAL_TYPE>(M, K, N, TA, TB, IDX_TFTYPE, NA, NW, LUT_VAL_TFTYPE)).Run(iters);   \
  }                                                                            \
  BENCHMARK(BM_MatMulLUT##_##M##_##K##_##N##_##TA##_##TB##_##IDX_TYPE##_##IDX_TFTYPE##_##NA##_##NW##__##LUT_VAL_TYPE##_##LUT_VAL_TFTYPE##_##DEVICE);

#define BM_MatMulLUT(M, K, N, TA, TB, NA, NW)                                       \
  BM_MatMulLUTDev(M, K, N, TA, TB, uint8, DT_UINT8, NA, NW, int32, DT_INT32, cpu);                   \
  BM_MatMulLUTDev(M, K, N, TA, TB, uint16, DT_UINT16, NA, NW, int32, DT_INT32, cpu);                   \
  BM_MatMulLUTDev(M, K, N, TA, TB, uint32, DT_UINT32, NA, NW, int32, DT_INT32, cpu);                   \
  BM_MatMulLUTDev(M, K, N, TA, TB, uint8, DT_UINT8, NA, NW, int64, DT_INT64, cpu);                   \
  BM_MatMulLUTDev(M, K, N, TA, TB, uint16, DT_UINT16, NA, NW, int64, DT_INT64, cpu);                   \
  BM_MatMulLUTDev(M, K, N, TA, TB, uint32, DT_UINT32, NA, NW, int64, DT_INT64, cpu);                   \

// Batch size of 1 included for inference.
// Typical fully connected layers
BM_MatMulLUT(1, 512, 512, false, false, 32, 1000);
BM_MatMulLUT(8, 512, 512, false, false, 32, 1000);
BM_MatMulLUT(16, 512, 512, false, false, 32, 1000);
BM_MatMulLUT(128, 512, 512, false, false, 32, 1000);

/* BM_MatMulLUT(1, 1024, 1024, false, false); */
/* BM_MatMulLUT(8, 1024, 1024, false, false); */
/* BM_MatMulLUT(16, 1024, 1024, false, false); */
/* BM_MatMulLUT(128, 1024, 1024, false, false); */
/* BM_MatMulLUT(4096, 4096, 4096, false, false); */

/* // Backward for fully connected layers */
/* BM_MatMulLUT(1, 1024, 1024, false, true); */
/* BM_MatMulLUT(8, 1024, 1024, false, true); */
/* BM_MatMulLUT(16, 1024, 1024, false, true); */
/* BM_MatMulLUT(128, 1024, 1024, false, true); */

/* // Forward softmax with large output size */
/* BM_MatMulLUT(1, 200, 10000, false, false); */
/* BM_MatMulLUT(8, 200, 10000, false, false); */
/* BM_MatMulLUT(20, 200, 10000, false, false); */
/* BM_MatMulLUT(20, 200, 20000, false, false); */

/* // Backward softmax with large output size */
/* BM_MatMulLUT(1, 10000, 200, false, true); */
/* BM_MatMulLUT(1, 10000, 200, false, false); */
/* BM_MatMulLUT(8, 10000, 200, false, true); */
/* BM_MatMulLUT(20, 10000, 200, false, true); */
/* BM_MatMulLUT(20, 20000, 200, false, true); */

/* // Test some matrix-vector multiplies. */
/* BM_MatMulLUT(50, 50, 1, false, false); */
/* BM_MatMulLUT(50, 50, 1, true, false); */
/* BM_MatMulLUT(50, 50, 1, false, true); */
/* BM_MatMulLUT(50, 50, 1, true, true); */
/* BM_MatMulLUT(500, 500, 1, false, false); */
/* BM_MatMulLUT(500, 500, 1, true, false); */
/* BM_MatMulLUT(500, 500, 1, false, true); */
/* BM_MatMulLUT(500, 500, 1, true, true); */
/* BM_MatMulLUT(2000, 2000, 1, false, false); */
/* BM_MatMulLUT(2000, 2000, 1, true, false); */
/* BM_MatMulLUT(2000, 2000, 1, false, true); */
/* BM_MatMulLUT(2000, 2000, 1, true, true); */

/* // Test some vector-matrix multiplies. */
/* BM_MatMulLUT(1, 50, 50, false, false); */
/* BM_MatMulLUT(1, 50, 50, true, false); */
/* BM_MatMulLUT(1, 50, 50, false, true); */
/* BM_MatMulLUT(1, 50, 50, true, true); */
/* BM_MatMulLUT(1, 500, 500, false, false); */
/* BM_MatMulLUT(1, 500, 500, true, false); */
/* BM_MatMulLUT(1, 500, 500, false, true); */
/* BM_MatMulLUT(1, 500, 500, true, true); */
/* BM_MatMulLUT(1, 2000, 2000, false, false); */
/* BM_MatMulLUT(1, 2000, 2000, true, false); */
/* BM_MatMulLUT(1, 2000, 2000, false, true); */
/* BM_MatMulLUT(1, 2000, 2000, true, true); */

/* // Test some rank-one products. */
/* BM_MatMulLUT(50, 1, 50, false, false); */
/* BM_MatMulLUT(50, 1, 50, true, false); */
/* BM_MatMulLUT(50, 1, 50, false, true); */
/* BM_MatMulLUT(50, 1, 50, true, true); */
/* BM_MatMulLUT(500, 1, 500, false, false); */
/* BM_MatMulLUT(500, 1, 500, true, false); */
/* BM_MatMulLUT(500, 1, 500, false, true); */
/* BM_MatMulLUT(500, 1, 500, true, true); */
/* BM_MatMulLUT(2000, 1, 2000, false, false); */
/* BM_MatMulLUT(2000, 1, 2000, true, false); */
/* BM_MatMulLUT(2000, 1, 2000, false, true); */
/* BM_MatMulLUT(2000, 1, 2000, true, true); */

} // tensorflow
