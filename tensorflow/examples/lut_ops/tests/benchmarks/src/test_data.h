#pragma once

#include <type_traits>

#include "defines.h"
#include "lookup_table.h"


//-----------------------------------------------------------------------------
// Test Data
//-----------------------------------------------------------------------------
struct TestData {

  TestData (const int32 n_activations,
            const int32 n_weights,
            const int32 n_batches,
            const int32 n_inputs,
            const int32 n_units)
    : n_activations (n_activations)
    , n_weights (n_weights)
    , n_batches (n_batches)
    , n_inputs (n_inputs)
    , n_units (n_units)
  {
    // generate random activations in [-1, 1]
    activations_f = VecF (n_activations);
    activations_f.setRandom ();
    activations_f = activations_f * 2.f - 1.f; 

    // generate random weights in [-1, 1]
    weights_f = VecF (n_weights);
    weights_f.setRandom ();
    weights_f = weights_f * 2.f - 1.f; 

    // generate lookup tables
    lut_f = createLookupTable (activations_f, weights_f);
    lut_i = (lut_f * scale).cast<int32> ();
    
    // generate random input data
    // activation indices in [0, n_activations]
    // weight indices in [0, n_weights]
    a_vals = MatF (n_batches, n_inputs);
    a_vals.setRandom ();
    a_idcs = (a_vals * static_cast<float> (n_activations)).cast<int32>();
    a_vals = lookupValues(a_idcs, activations_f);

    w_vals = MatF (n_inputs, n_units);
    w_vals.setRandom ();
    w_idcs = (w_vals * static_cast<float> (n_weights)).cast<int32>();
    w_vals = lookupValues(w_idcs, weights_f);

    product_dims[0].first = 1;
    product_dims[0].second = 0;
  }

  /* //--------------------------------------------------------------------------- */
  /* template <typename T> */
  /* Mat<T> get_act () const { */
  /*   return std::is_floating_point<T>::value ? a_vals : a_idcs;}; */
  /* //--------------------------------------------------------------------------- */
  /* template <typename T> */
  /* Mat<T> get_weights () const { */
  /*   return std::is_floating_point<T>::value ? w_vals : w_idcs;}; */

  //---------------------------------------------------------------------------
  VecF activations_f;
  VecF weights_f;

  MatI a_idcs;
  MatF a_vals;
  MatI w_idcs;
  MatF w_vals;
  
  MatF lut_f;
  MatI lut_i;
  
  using DimPair = Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>;
  DimPair product_dims;

  const int32 n_activations {0};
  const int32 n_weights {0};
  const int32 n_batches {0};
  const int32 n_inputs {0};
  const int32 n_units {0};

  const float scale {16384.f}; // 2**14
};
//-----------------------------------------------------------------------------

