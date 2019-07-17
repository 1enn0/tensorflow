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

template <typename T>
static Graph* MatMulNaive(int m, int k, int n, bool transpose_a, bool transpose_b,
                     DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();
  
  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("MatMulNaive"), "MatMulNaive")
                  .Input(test::graph::Constant(g, in0))
                  .Input(test::graph::Constant(g, in1))
                  .Attr("transpose_a", transpose_a)
                  .Attr("transpose_b", transpose_b)
                  .Finalize(g, &node));
  return g;
}

#define BM_MatMulNaiveDev(M, K, N, TA, TB, T, TFTYPE, DEVICE)                       \
  static void BM_MatMulNaive##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE( \
      int iters) {                                                             \
    testing::UseRealTime();                                                    \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);        \
    test::Benchmark(#DEVICE, MatMulNaive<T>(M, K, N, TA, TB, TFTYPE)).Run(iters);   \
  }                                                                            \
  BENCHMARK(BM_MatMulNaive##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE);

#define BM_MatMulNaive(M, K, N, TA, TB)                                       \
  BM_MatMulNaiveDev(M, K, N, TA, TB, float, DT_FLOAT, cpu);                   \
  /* BM_MatMulNaiveDev(M, K, N, TA, TB, float, DT_FLOAT, gpu);                   \ */
  /* BM_MatMulNaiveDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, cpu); \ */
  /* BM_MatMulNaiveDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, gpu); \ */

// Batch size of 1 included for inference.
// Typical fully connected layers
BM_MatMulNaive(1, 512, 512, false, false);
BM_MatMulNaive(8, 512, 512, false, false);
BM_MatMulNaive(16, 512, 512, false, false);
BM_MatMulNaive(128, 512, 512, false, false);

BM_MatMulNaive(1, 1024, 1024, false, false);
BM_MatMulNaive(8, 1024, 1024, false, false);
BM_MatMulNaive(16, 1024, 1024, false, false);
BM_MatMulNaive(128, 1024, 1024, false, false);
BM_MatMulNaive(4096, 4096, 4096, false, false);

// Backward for fully connected layers
BM_MatMulNaive(1, 1024, 1024, false, true);
BM_MatMulNaive(8, 1024, 1024, false, true);
BM_MatMulNaive(16, 1024, 1024, false, true);
BM_MatMulNaive(128, 1024, 1024, false, true);

// Forward softmax with large output size
BM_MatMulNaive(1, 200, 10000, false, false);
BM_MatMulNaive(8, 200, 10000, false, false);
BM_MatMulNaive(20, 200, 10000, false, false);
BM_MatMulNaive(20, 200, 20000, false, false);

// Backward softmax with large output size
BM_MatMulNaive(1, 10000, 200, false, true); BM_MatMulNaive(1, 10000, 200, false, false); BM_MatMulNaive(8, 10000, 200, false, true);
BM_MatMulNaive(20, 10000, 200, false, true);
BM_MatMulNaive(20, 20000, 200, false, true);

// Test some matrix-vector multiplies.
BM_MatMulNaive(50, 50, 1, false, false);
BM_MatMulNaive(50, 50, 1, true, false);
BM_MatMulNaive(50, 50, 1, false, true);
BM_MatMulNaive(50, 50, 1, true, true);
BM_MatMulNaive(500, 500, 1, false, false);
BM_MatMulNaive(500, 500, 1, true, false);
BM_MatMulNaive(500, 500, 1, false, true);
BM_MatMulNaive(500, 500, 1, true, true);
BM_MatMulNaive(2000, 2000, 1, false, false);
BM_MatMulNaive(2000, 2000, 1, true, false);
BM_MatMulNaive(2000, 2000, 1, false, true);
BM_MatMulNaive(2000, 2000, 1, true, true);

// Test some vector-matrix multiplies.
BM_MatMulNaive(1, 50, 50, false, false);
BM_MatMulNaive(1, 50, 50, true, false);
BM_MatMulNaive(1, 50, 50, false, true);
BM_MatMulNaive(1, 50, 50, true, true);
BM_MatMulNaive(1, 500, 500, false, false);
BM_MatMulNaive(1, 500, 500, true, false);
BM_MatMulNaive(1, 500, 500, false, true);
BM_MatMulNaive(1, 500, 500, true, true);
BM_MatMulNaive(1, 2000, 2000, false, false);
BM_MatMulNaive(1, 2000, 2000, true, false);
BM_MatMulNaive(1, 2000, 2000, false, true);
BM_MatMulNaive(1, 2000, 2000, true, true);

// Test some rank-one products.
BM_MatMulNaive(50, 1, 50, false, false);
BM_MatMulNaive(50, 1, 50, true, false);
BM_MatMulNaive(50, 1, 50, false, true);
BM_MatMulNaive(50, 1, 50, true, true);
BM_MatMulNaive(500, 1, 500, false, false);
BM_MatMulNaive(500, 1, 500, true, false);
BM_MatMulNaive(500, 1, 500, false, true);
BM_MatMulNaive(500, 1, 500, true, true);
BM_MatMulNaive(2000, 1, 2000, false, false);
BM_MatMulNaive(2000, 1, 2000, true, false);
BM_MatMulNaive(2000, 1, 2000, false, true);
BM_MatMulNaive(2000, 1, 2000, true, true);

} // tensorflow
