package(
    default_visibility = [
        "//visibility:public",
    ],
)

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")


tf_custom_op_library(
    name = "my_composite_op.so",
    srcs = [
        "my_composite_op.cc",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/kernels:matmul_op",
    ],
)

#------------------------------------------------------------------------------

