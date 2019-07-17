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
load("//tensorflow:tensorflow.bzl", "tf_py_test")


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

#------------------------------------------------------------------------------
# TESTS
#------------------------------------------------------------------------------
COMMON_TEST_DEPS = [
        "//tensorflow/core:user_ops_op_lib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
]

tf_cc_test(
    name = "matmul_naive_op_test",
    size = "small",
    srcs = ["matmul_naive_op_test.cc"],
    tags = ["nomsan"],  # http://b/32242946
    deps = COMMON_TEST_DEPS,
)

tf_cc_test(
    name = "matmul_lut_op_test",
    size = "small",
    srcs = ["matmul_lut_op_test.cc"],
    tags = ["nomsan"],  # http://b/32242946
    deps = COMMON_TEST_DEPS,
)

tf_py_test(
    name = "my_matmul_benchmark",
    size = "medium",
    srcs = ["my_matmul_benchmark.py"],
    additional_deps = [
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    main = "my_matmul_benchmark.py",
)

#------------------------------------------------------------------------------
# BENCHMARKS
#------------------------------------------------------------------------------
tf_cc_test(
    name = "matmul_naive_op_benchmark",
    size = "small",
    srcs = ["matmul_naive_op_benchmark.cc"],
    tags = ["nomsan"],  # http://b/32242946
    deps = COMMON_TEST_DEPS,
)

tf_cc_test(
    name = "matmul_lut_op_benchmark",
    size = "small",
    srcs = ["matmul_lut_op_benchmark.cc"],
    tags = ["nomsan"],  # http://b/32242946
    deps = COMMON_TEST_DEPS,
)
