#include <type_traits>

#include "benchmark/benchmark.h"

#include "defines.h"
#include "test_data.h"
#include "lookup_table.h"

#include "eigen_matmul_naive-inl.h"
#include "eigen_matmul_lut-inl.h"

//-----------------------------------------------------------------------------
// MatMul Benchmarks (Using Eigen Contraction Implementation)
//
// n_activations / n_weights are fixed, as they do not have any
// impact on the performance (no LUT is used)
//-----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Eigen_Int32(benchmark::State& state) {
  auto data = TestData<int32>(32, 1000, n_batches, n_inputs, n_units);
  for (auto _ : state)
    MatI product = data.a_idcs.contract(data.w_idcs, data.product_dims);
}

//-----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Eigen_Float(benchmark::State& state) {
  auto data = TestData<int32>(32, 1000, n_batches, n_inputs, n_units);
  for (auto _ : state)
    MatF product = data.a_vals.contract(data.w_vals, data.product_dims);
}

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Eigen_Int32, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int32, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int32, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int32, 32, 10000, 64);
/* BENCHMARK_TEMPLATE(BM_Eigen_Int32, 32, 100000, 1024); */

BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 10000, 64);
/* BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 100000, 1024); */




//-----------------------------------------------------------------------------
// MatMul Benchmarks (Naive Implementation)
//
// n_activations / n_weights are fixed, as they do not have any
// impact on the performance (no LUT is used)
//-----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Naive_Int32(benchmark::State& state) {
  auto data = TestData<int32>(32, 1000, n_batches, n_inputs, n_units);
  for (auto _ : state)
    MatI product = Eigen::MatMulNaive(data.a_idcs, data.w_idcs, data.product_dims);
}

//-----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Naive_Int_Col(benchmark::State& state) {
  auto data = TestData<int32>(32, 1000, n_batches, n_inputs, n_units);
  Mat<int32, Eigen::ColMajor> w_idcs_col_major = data.w_idcs.swap_layout();
  data.product_dims[0].second = 1;
  for (auto _ : state)
    MatI result = Eigen::MatMulNaive(data.a_idcs, w_idcs_col_major, data.product_dims);
}

//-----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Naive_Float(benchmark::State& state) {
  auto data = TestData<int32>(32, 1000, n_batches, n_inputs, n_units);
  for (auto _ : state)
    MatF product = Eigen::MatMulNaive(data.a_vals, data.w_vals, data.product_dims);
}

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Naive_Int32, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int32, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int32, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int32, 32, 10000, 64);
/* BENCHMARK_TEMPLATE(BM_Naive_Int32, 32, 100000, 1024); */

BENCHMARK_TEMPLATE(BM_Naive_Int_Col, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int_Col, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int_Col, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int_Col, 32, 10000, 64);
/* BENCHMARK_TEMPLATE(BM_Naive_Int_Col, 32, 100000, 1024); */

BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 10000, 64);
/* BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 100000, 1024); */




//-----------------------------------------------------------------------------
// MatMulLut Benchmarks
//-----------------------------------------------------------------------------
template <int32 n_act, int32 n_weights, int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Lut_Int32(benchmark::State& state) {
  auto data = TestData<int32>(n_act, n_weights, n_batches, n_inputs, n_units); 
  for (auto _ : state)
    MatI product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_i, data.product_dims);
}

template <int32 n_act, int32 n_weights, int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Lut_V2_Int32(benchmark::State& state) {
  auto data = TestData<int32>(n_act, n_weights, n_batches, n_inputs, n_units); 
  for (auto _ : state)
    MatI product = Eigen::MatMulLut_V2(data.a_idcs, data.w_idcs, data.lut_i, data.product_dims);
}

//-----------------------------------------------------------------------------
template <int32 n_act, int32 n_weights, int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Lut_Float(benchmark::State& state) { 
  auto data = TestData<int32>(n_act, n_weights, n_batches, n_inputs, n_units); 
  for (auto _ : state) 
    MatF product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_f, data.product_dims); 
} 

//-----------------------------------------------------------------------------
// SMALL LUT [32 x 100] // 12.8K size -> fits into 32K L1 cache
//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_Int32, 32, 100, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 32, 100, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 32, 100, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 32, 100, 32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 32, 100, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 32, 100, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 32, 100, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 32, 100, 32, 10000, 64);
/* BENCHMARK_TEMPLATE(BM_Lut_Int32, 32, 100, 32, 100000, 1024); */


/* BENCHMARK_TEMPLATE(BM_Lut_Float, 32, 100, 32, 10, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 32, 100, 32, 100, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 32, 100, 32, 1000, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 32, 100, 32, 10000, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 32, 100, 32, 100000, 1024); */

//-----------------------------------------------------------------------------
// MEDIUM LUT [32 x 1000] // 128K size -> fits into 256K L2 cache
//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_Int32, 32, 1000, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 32, 1000, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 32, 1000, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 32, 1000, 32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 32, 1000, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 32, 1000, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 32, 1000, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 32, 1000, 32, 10000, 64);
/* BENCHMARK_TEMPLATE(BM_Lut_Int32, 32, 1000, 32, 100000, 1024); */

/* BENCHMARK_TEMPLATE(BM_Lut_Float, 32, 1000, 32, 10, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 32, 1000, 32, 100, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 32, 1000, 32, 1000, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 32, 1000, 32, 10000, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 32, 1000, 32, 100000, 1024); */

//-----------------------------------------------------------------------------
// LARGE LUT [64 x 10000] // 2560K size -> fits into 4096K L3 cache
//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_Int32, 64, 10000, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 64, 10000, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 64, 10000, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 64, 10000, 32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 64, 10000, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 64, 10000, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 64, 10000, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 64, 10000, 32, 10000, 64);
/* BENCHMARK_TEMPLATE(BM_Lut_Int32, 64, 10000, 32, 100000, 1024); */

/* BENCHMARK_TEMPLATE(BM_Lut_Float, 64, 10000, 32, 10, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 64, 10000, 32, 100, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 64, 10000, 32, 1000, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 64, 10000, 32, 10000, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 64, 10000, 32, 100000, 64); */

//-----------------------------------------------------------------------------
// XLARGE LUT [256 x 10000] // 10240K size -> does not fit any cache
//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_Int32, 256, 10000, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 256, 10000, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 256, 10000, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_Int32, 256, 10000, 32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 256, 10000, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 256, 10000, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 256, 10000, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_V2_Int32, 256, 10000, 32, 10000, 64);
/* BENCHMARK_TEMPLATE(BM_Lut_Int32, 256, 10000, 32, 100000, 1024); */

/* BENCHMARK_TEMPLATE(BM_Lut_Float, 256, 10000, 32, 10, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 256, 10000, 32, 100, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 256, 10000, 32, 1000, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 256, 10000, 32, 10000, 64); */
/* BENCHMARK_TEMPLATE(BM_Lut_Float, 256, 10000, 32, 100000, 1024); */

//-----------------------------------------------------------------------------
BENCHMARK_MAIN();
//-----------------------------------------------------------------------------
