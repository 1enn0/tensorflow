#include <type_traits>

#include "benchmark/benchmark.h"

#include "defines.h"
#include "test_data.h"
#include "lookup_table.h"

#include "eigen_matmul_naive-inl.h"
#include "eigen_matmul_lut-inl.h"

//-----------------------------------------------------------------------------
template <int32 n_activations, int32 n_weights, int32 n_batches,
          int32 n_inputs, int32 n_units> 
void BM_Eigen_Int(benchmark::State& state) {
  auto data = TestData(n_activations, n_weights, n_batches,
                       n_inputs, n_units);
  for (auto _ : state) {
    MatI product = data.a_idcs.contract(data.w_idcs, data.product_dims);
  }
}

//-----------------------------------------------------------------------------
template <int32 n_activations, int32 n_weights, int32 n_batches,
          int32 n_inputs, int32 n_units> 
void BM_Eigen_Float(benchmark::State& state) {
  auto data = TestData(n_activations, n_weights, n_batches,
                       n_inputs, n_units);
  for (auto _ : state) {
    MatF product = data.a_vals.contract(data.w_vals, data.product_dims);
  }
}

//-----------------------------------------------------------------------------
template <int32 n_activations, int32 n_weights, int32 n_batches,
          int32 n_inputs, int32 n_units> 
void BM_Naive_Int(benchmark::State& state) {
  auto data = TestData(n_activations, n_weights, n_batches,
                       n_inputs, n_units);
  for (auto _ : state) {
    MatI product = Eigen::MatMulNaive(data.a_idcs, data.w_idcs, data.product_dims);
  }
}

//-----------------------------------------------------------------------------
template <int32 n_activations, int32 n_weights, int32 n_batches,
          int32 n_inputs, int32 n_units> 
void BM_Naive_Float(benchmark::State& state) {
  auto data = TestData(n_activations, n_weights, n_batches,
                       n_inputs, n_units);
  for (auto _ : state) {
    MatF product = Eigen::MatMulNaive(data.a_vals, data.w_vals, data.product_dims);
  }
}

//-----------------------------------------------------------------------------
template <int32 n_activations, int32 n_weights, int32 n_batches,
          int32 n_inputs, int32 n_units> 
void BM_Lut_Int(benchmark::State& state) {
  auto data = TestData(n_activations, n_weights, n_batches,
                       n_inputs, n_units);
  for (auto _ : state) {
    MatI product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_i, data.product_dims);
  }
}

//-----------------------------------------------------------------------------
template <int32 n_activations, int32 n_weights, int32 n_batches,
          int32 n_inputs, int32 n_units> 
void BM_Lut_Float(benchmark::State& state) {
  auto data = TestData(n_activations, n_weights, n_batches,
                       n_inputs, n_units);
  for (auto _ : state) {
    MatF product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_f, data.product_dims);
  }
}

//-----------------------------------------------------------------------------
// Benchmarks
//-----------------------------------------------------------------------------

BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 1000, 1, 10, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 1000, 1, 100, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 1000, 1, 1000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 1000, 1, 10000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 1000, 1, 100000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 1000, 1, 1000000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 1000, 1, 10000000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 1000, 1, 10, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 1000, 1, 100, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 1000, 1, 1000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 1000, 1, 10000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 1000, 1, 100000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 1000, 1, 1000000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 1000, 1, 10000000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 1000, 1, 10, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 1000, 1, 100, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 1000, 1, 1000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 1000, 1, 10000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 1000, 1, 100000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 1000, 1, 1000000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 1000, 1, 10000000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 1000, 1, 10, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 1000, 1, 100, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 1000, 1, 1000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 1000, 1, 10000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 1000, 1, 100000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 1000, 1, 1000000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 1000, 1, 10000000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_Int,     32, 1000, 1, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int,     32, 1000, 1, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int,     32, 1000, 1, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int,     32, 1000, 1, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int,     32, 1000, 1, 100000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int,     32, 1000, 1, 1000000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int,     32, 1000, 1, 10000000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_Float,   32, 1000, 1, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_Float,   32, 1000, 1, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_Float,   32, 1000, 1, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Float,   32, 1000, 1, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Float,   32, 1000, 1, 100000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Float,   32, 1000, 1, 1000000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Float,   32, 1000, 1, 10000000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_MAIN();
