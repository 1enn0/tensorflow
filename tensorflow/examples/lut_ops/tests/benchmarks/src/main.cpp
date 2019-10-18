#include <iostream>

#include "test_data.h"

#include "eigen_matmul_naive-inl.h"
#include "eigen_matmul_lut-inl.h"

int main ()
{
  using namespace Eigen;

  /* Tensor<int, 1, RowMajor, DenseIndex> l (16); */
  /* l = l.setConstant(1).cumsum(0); */

  /* Tensor<int, 2, RowMajor, DenseIndex> r (4, 4); */
  /* Tensor<int, 2, ColMajor, DenseIndex> c (4, 4); */
  
  /* array<int, 2> two_dims {{4, 4}}; */
  /* array<int, 2> shuffle {{1, 0}}; */
  /* r = l.reshape(two_dims); */
  /* c = r.swap_layout().shuffle(shuffle); */

  /* std::cout << r << "\n\n"; */
  /* std::cout << c << "\n\n"; */
  /* std::cout << range << "\n\n" */


  /* auto data = TestData<int>(32, 1000, 6, 100, 6); */

  /* MatI product1 = MatMulLut(data.a_idcs, data.w_idcs, data.lut_i, data.product_dims); */
  /* MatI product2 = MatMulLut_V2(data.a_idcs, data.w_idcs, data.lut_i, data.product_dims); */

  /* std::cout << product1 << "\n\n"; */
  /* std::cout << product2 << "\n\n"; */

  Tensor<int, 3, RowMajor, DenseIndex> r (4, 5, 6);
  r.setConstant(1);

  array<int, 1> sum_dim {2};
  Tensor<int, 2, RowMajor, DenseIndex> rs = r.sum(sum_dim);

  std::cout << rs << std::endl;


  return 0;
}
