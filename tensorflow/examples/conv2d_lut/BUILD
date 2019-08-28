# Description:
# Custom OP library for conv2d using lookup table

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.bzl", "tf_exec_compatible_with")
load("//tensorflow:tensorflow.bzl", "tf_copts")

exports_files(["LICENSE"])

cc_library(
    name = "my_custom_op_lib",
    srcs = [
	"conv2d_lut_op.cc",
	"conv2d_lut_op_kernel.h",
	"conv2d_lut_op_kernel.cc",
    ],
    deps = [
	"//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/kernels:ops_util",
	"//tensorflow/core/kernels:conv_2d_hdrs",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    alwayslink = 1,
)


tf_custom_op_library(
    name = "conv2d_lut_op_kernel.so",
    srcs = [
	"conv2d_lut_op.cc",
	"conv2d_lut_op_kernel.h",
	"conv2d_lut_op_kernel.cc",
    ],
    deps = [
	"//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/kernels:ops_util",
	"//tensorflow/core/kernels:conv_2d_hdrs",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

py_library(
    name = "conv2d_lut_op",
    srcs = ["conv2d_lut_op.py"],
    data = [":conv2d_lut_op_kernel.so"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)
