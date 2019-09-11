project(my_eigen_benchmarks)
cmake_minimum_required(VERSION 3.14)

find_package(PkgConfig REQUIRED)

pkg_check_modules(BM REQUIRED benchmark)

set(EIGEN_INCLUDE_DIR 
${PROJECT_SOURCE_DIR}/../../../../../bazel-tensorflow/external/eigen_archive
)

#------------------------------------------------------------------------------
# Benchmark
#------------------------------------------------------------------------------
add_executable(bench "")
target_sources(bench
PRIVATE
  src/benchmarks.cpp
)
target_include_directories(bench
PRIVATE
  ${PROJECT_SOURCE_DIR}/../../kernels
  ${EIGEN_INCLUDE_DIR}
)

target_link_libraries(bench
PRIVATE
  ${BM_LIBRARIES}
)

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
add_executable(main "")
target_sources(main
PRIVATE
  src/main.cpp
)
target_include_directories(main
PRIVATE
  ${EIGEN_INCLUDE_DIR}
)
