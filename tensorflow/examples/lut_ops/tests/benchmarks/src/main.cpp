#include <iostream>

#include "test_data.h"

int main ()
{
  auto data = TestData(256, 4096, 32, 1000, 64);

  std::cout << data.a_idcs.minimum() << "\n\n";
  std::cout << data.a_idcs.maximum() << "\n\n";
  std::cout << data.w_idcs.minimum() << "\n\n";
  std::cout << data.w_idcs.maximum() << "\n\n";
  return 0;
}
