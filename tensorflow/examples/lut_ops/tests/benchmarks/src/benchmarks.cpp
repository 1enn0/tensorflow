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
void BM_Eigen_Int(benchmark::State& state) {
  auto data = TestData(32, 1000, n_batches, n_inputs, n_units);
  for (auto _ : state)
    MatI product = data.a_idcs.contract(data.w_idcs, data.product_dims);
}

//-----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Eigen_Float(benchmark::State& state) {
  auto data = TestData(32, 1000, n_batches, n_inputs, n_units);
  for (auto _ : state)
    MatF product = data.a_vals.contract(data.w_vals, data.product_dims);
}


//-----------------------------------------------------------------------------
// MatMul Benchmarks (Naive Implementation)
//
// n_activations / n_weights are fixed, as they do not have any
// impact on the performance (no LUT is used)
//-----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Naive_Int(benchmark::State& state) {
  auto data = TestData(32, 1000, n_batches, n_inputs, n_units);
  for (auto _ : state)
    MatI product = Eigen::MatMulNaive(data.a_idcs, data.w_idcs, data.product_dims);
}

//-----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Naive_Float(benchmark::State& state) {
  auto data = TestData(32, 1000, n_batches, n_inputs, n_units);
  for (auto _ : state)
    MatF product = Eigen::MatMulNaive(data.a_vals, data.w_vals, data.product_dims);
}


//-----------------------------------------------------------------------------
// MatMulLut Benchmarks with different LUT sizes
//-----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units> 
void BM_Lut_S_Int(benchmark::State& state) {
  auto data = TestData(6, 10, n_batches, n_inputs, n_units);
  for (auto _ : state)
    MatI product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_i, data.product_dims);
}

//-----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units>  
void BM_Lut_S_Float(benchmark::State& state) { 
  auto data = TestData(6, 10, n_batches, n_inputs, n_units); 
  for (auto _ : state) 
    MatF product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_f, data.product_dims); 
} 

//----------------------------------------------------------------------------- 
template <int32 n_batches, int32 n_inputs, int32 n_units>  
void BM_Lut_M_Int(benchmark::State& state) { 
  auto data = TestData(16, 100, n_batches, n_inputs, n_units); 
  for (auto _ : state)
    MatI product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_i, data.product_dims); 
} 

//----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units>  
void BM_Lut_M_Float(benchmark::State& state) { 
  auto data = TestData(16, 100, n_batches, n_inputs, n_units); 
  for (auto _ : state)
    MatF product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_f, data.product_dims); 
} 

//----------------------------------------------------------------------------- 
template <int32 n_batches, int32 n_inputs, int32 n_units>  
void BM_Lut_L_Int(benchmark::State& state) { 
  auto data = TestData(32, 1000, n_batches, n_inputs, n_units); 
  for (auto _ : state)
    MatI product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_i, data.product_dims); 
} 

//----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units>  
void BM_Lut_L_Float(benchmark::State& state) { 
  auto data = TestData(32, 1000, n_batches, n_inputs, n_units); 
  for (auto _ : state)
    MatF product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_f, data.product_dims); 
} 

//----------------------------------------------------------------------------- 
template <int32 n_batches, int32 n_inputs, int32 n_units>  
void BM_Lut_XL_Int(benchmark::State& state) { 
  auto data = TestData(256, 4096, n_batches, n_inputs, n_units); 
  for (auto _ : state)
    MatI product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_i, data.product_dims); 
} 

//----------------------------------------------------------------------------
template <int32 n_batches, int32 n_inputs, int32 n_units>  
void BM_Lut_XL_Float(benchmark::State& state) { 
  auto data = TestData(256, 4096, n_batches, n_inputs, n_units); 
  for (auto _ : state)
    MatF product = Eigen::MatMulLut(data.a_idcs, data.w_idcs, data.lut_f, data.product_dims); 
} 

//-----------------------------------------------------------------------------
// Benchmarks
//-----------------------------------------------------------------------------

BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 10, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 100, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Int,   32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Eigen_Float, 32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 10, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 100, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Int,   32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 10, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 100, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Naive_Float, 32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_S_Int,     32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_S_Int,     32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_S_Int,     32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_S_Int,     32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_S_Int,     32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_S_Float,   32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_S_Float,   32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_S_Float,   32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_S_Float,   32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_S_Float,   32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_M_Int,     32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_M_Int,     32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_M_Int,     32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_M_Int,     32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_M_Int,     32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_M_Float,   32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_M_Float,   32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_M_Float,   32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_M_Float,   32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_M_Float,   32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_L_Int,     32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_L_Int,     32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_L_Int,     32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_L_Int,     32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_L_Int,     32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_L_Float,   32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_L_Float,   32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_L_Float,   32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_L_Float,   32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_L_Float,   32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_XL_Int,     32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_XL_Int,     32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_XL_Int,     32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_XL_Int,     32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_XL_Int,     32, 100000, 64);

//-----------------------------------------------------------------------------
BENCHMARK_TEMPLATE(BM_Lut_XL_Float,   32, 10, 64);
BENCHMARK_TEMPLATE(BM_Lut_XL_Float,   32, 100, 64);
BENCHMARK_TEMPLATE(BM_Lut_XL_Float,   32, 1000, 64);
BENCHMARK_TEMPLATE(BM_Lut_XL_Float,   32, 10000, 64);
BENCHMARK_TEMPLATE(BM_Lut_XL_Float,   32, 100000, 64);






BENCHMARK_MAIN();
