#pragma once

#include <iostream>
#include <string>

#include "defines.h"

//-----------------------------------------------------------------------------
template <typename T>
Mat<T> createLookupTable(const Vec<T>& activations, const Vec<T>& weights)
{
  Mat<T> lut (activations.size() + 1, weights.size() + 1);
  for (int32 i {0}; i < activations.size(); ++i)
    for (int32 j {0}; j < weights.size(); ++j)
      lut(i, j) = activations(i) * weights(j);

  for (int32 i {0}; i < activations.size(); ++i)
    lut(i, weights.size()) = activations(i);

  for (int32 j {0}; j < weights.size(); ++j)
    lut(activations.size(), j) = weights(j);

  return lut;
}


//-----------------------------------------------------------------------------
template <typename T, typename U>
Mat<U> lookupValues(const Mat<T>& matIndices, const Vec<U>& values)
{
  const int32 nRows = matIndices.dimension(0);
  const int32 nCols = matIndices.dimension(1);
  Mat<U> result (nRows, nCols);

  for (int32 i {0}; i < nRows; ++i)
    for (int32 j {0}; j < nCols; ++j)
      result(i, j) = values(matIndices(i, j));

  return result; 
}

//-----------------------------------------------------------------------------
template <typename T>
void printDimensions(const T& t, const std::string& name)
{
  std::cout << name << ": [";
  for (int32 i {0}; i < t.NumDimensions - 1; ++i)
    std::cout << t.dimension(i) << ", ";  
  std::cout << t.dimension(t.NumDimensions - 1) << "]\n";  
}

