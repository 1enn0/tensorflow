# Description:
# Custom OP library for conv2d/matmul using lookup table

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_exec_compatible_with")

exports_files(["LICENSE"])

tf_custom_op_library(
    name = "lut_ops_op_lib.so",
    srcs = [
      "ops/conv2d_lut_op.cc",
      "kernels/conv2d_lut_op_kernel.h",
      "kernels/conv2d_lut_op_kernel.cc",
      "kernels/conv2d_lut.h",
      "kernels/eigen_spatial_convolutions_lut-inl.h",
      "ops/matmul_lut_op.cc",
      "kernels/matmul_lut_op_kernel.h",
      "kernels/matmul_lut_op_kernel.cc",
      "kernels/eigen_matmul_lut-inl.h",
      "ops/matmul_naive_op.cc",
      "kernels/matmul_naive_op_kernel.h",
      "kernels/matmul_naive_op_kernel.cc",
      "kernels/eigen_matmul_naive-inl.h",
    ],
    deps = [
      "//tensorflow/core/kernels:eigen_helpers",
      "//tensorflow/core/kernels:ops_util",
      "//tensorflow/core/kernels:fill_functor",
      "//tensorflow/core:core_cpu",
      "//tensorflow/core:framework",
      "//tensorflow/core:lib",
      "//tensorflow/core:lib_internal",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "lut_ops",
    srcs = ["lut_ops.py"],
    data = [":lut_ops_op_lib.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)
