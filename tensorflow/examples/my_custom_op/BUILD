package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "my_custom_op",
    srcs = [
        "main.cc",
    ],
    deps = [
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:tensorflow",
    ],
)
