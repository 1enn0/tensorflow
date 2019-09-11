#include <iostream>

#include "test_data.h"

int main ()
{
  auto data = TestData(10, 10, 2, 5, 3);

  std::cout << data.activations_f << "\n\n";
  std::cout << data.weights_f << "\n\n";
  std::cout << data.a_idcs << "\n\n";
  std::cout << data.a_vals << "\n\n";
  std::cout << data.w_idcs << "\n\n";
  std::cout << data.w_vals << "\n\n";
  /* std::cout << data.lut_i << "\n\n"; */
  /* std::cout << data.lut_f << "\n\n"; */
  return 0;
}
