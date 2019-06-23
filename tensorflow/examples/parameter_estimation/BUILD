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
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "estimate_params_from_wav_lib",
    srcs = ["estimate_params_from_wav.cc"],
    hdrs = ["estimate_params_from_wav.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_binary(
    name = "estimate_params_from_wav",
    srcs = ["main.cc"],
    deps = [
        ":estimate_params_from_wav_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)