# Description:
# Custom OP library for conv2d using lookup table

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_exec_compatible_with")

exports_files(["LICENSE"])

tf_custom_op_library(
    name = "conv2d_lut_op_kernel.so",
    srcs = [
	    "ops/conv2d_lut_op.cc",
	    "kernels/conv2d_lut_op_kernel.h",
	    "kernels/conv2d_lut_op_kernel.cc",
      "kernels/conv2d_lut.h",
      "kernels/eigen_spatial_convolutions_lut-inl.h",
    ],
    deps = [
	    "//tensorflow/core/kernels:eigen_helpers",
      "//tensorflow/core/kernels:ops_util",
      "//tensorflow/core:core_cpu",
      "//tensorflow/core:framework",
      "//tensorflow/core:lib",
      "//tensorflow/core:lib_internal",
    ],
)

py_library(
    name = "conv2d_lut_op",
    srcs = ["conv2d_lut_op.py"],
    data = [":conv2d_lut_op_kernel.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)
