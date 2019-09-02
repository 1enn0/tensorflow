# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test for MatMulLUT op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os.path

import tensorflow as tf

import numpy as np

matmul_lut = None

class MatMulLutTest(unittest.TestCase):

    def setUp(self):
        self.n_activations = 6
        self.n_weights = 10

        # low, high values here are arbitrary
        self.activations = np.random.randint(0, 100, size=self.n_activations)
        self.weights = np.random.randint(5, 15, size=self.n_weights)

        # lut shape [n_activations + 1, n_weights + 1]
        self.lut = np.zeros([self.n_activations + 1, self.n_weights + 1], dtype=np.int32)
        self.lut[:-1, :-1] = np.array(np.expand_dims(self.activations, 1) * np.expand_dims(self.weights, 0))
        self.lut[-1, :-1] = self.weights
        self.lut[:-1, -1] = self.activations

        self.lut_f32 = self.lut.astype(np.float32)

    def create_inputs(self, activations_shape, weights_shape):
        """Create input arrays.

        Args:
            activations_shape: [n_batches, n_inputs]
            weights_shape: [n_inputs, n_units]

        Returns:
            the two arrays activations, weights
        """
        assert activations_shape[1] == weights_shape[0]
        inputs = np.random.randint(
            0, self.n_activations, size=activations_shape).astype(np.int32)
        kernels = np.random.randint(
            0, self.n_weights, size=weights_shape).astype(np.int32)

        return inputs, kernels

    def compute_expected(self, input_idcs, kernel_idcs):
        """Compute expected output with regular conv2d function.

        For this, we look up the values corresponding
        to the input index arrays and cast them
        to float32 (as conv2d does not support int32) and
        cast the result back to int32.
        """
        activations_values = self.activations[input_idcs]
        weights_values = self.weights[kernel_idcs]

        expected = tf.matmul(activations_values, weights_values)
        return expected.numpy()

    def test_batch1_inputs100_units6_int32(self):
        """ test names have format:
        test_batch[number of batches]_inputs[number of inputs]_units[number of units]_[lookup table dtype]
        """
        act_idcs, w_idcs = self.create_inputs([1, 100], [100, 6])
        result = matmul_lut(act_idcs, w_idcs, self.lut).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs100_units6_float32(self):
        act_idcs, w_idcs = self.create_inputs([1, 100], [100, 6])
        result = matmul_lut(act_idcs, w_idcs, self.lut_f32).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch10_inputs100_units64_int32(self):
        act_idcs, w_idcs = self.create_inputs([10, 100], [100, 64])
        result = matmul_lut(act_idcs, w_idcs, self.lut).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch10_inputs100_units64_float32(self):
        act_idcs, w_idcs = self.create_inputs([10, 100], [100, 64])
        result = matmul_lut(act_idcs, w_idcs, self.lut_f32).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs1000_units64_int32(self):
        act_idcs, w_idcs = self.create_inputs([1, 1000], [1000, 64])
        result = matmul_lut(act_idcs, w_idcs, self.lut).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs1000_units64_float32(self):
        act_idcs, w_idcs = self.create_inputs([1, 1000], [1000, 64])
        result = matmul_lut(act_idcs, w_idcs, self.lut_f32).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs10000_units64_int32(self):
        act_idcs, w_idcs = self.create_inputs([1, 10000], [10000, 64])
        result = matmul_lut(act_idcs, w_idcs, self.lut).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs10000_units64_float32(self):
        act_idcs, w_idcs = self.create_inputs([1, 10000], [10000, 64])
        result = matmul_lut(act_idcs, w_idcs, self.lut_f32).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs1000_units256_int32(self):
        act_idcs, w_idcs = self.create_inputs([1, 1000], [1000, 256])
        result = matmul_lut(act_idcs, w_idcs, self.lut).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs1000_units256_float32(self):
        act_idcs, w_idcs = self.create_inputs([1, 1000], [1000, 256])
        result = matmul_lut(act_idcs, w_idcs, self.lut_f32).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs1000_units512_int32(self):
        act_idcs, w_idcs = self.create_inputs([1, 1000], [1000, 512])
        result = matmul_lut(act_idcs, w_idcs, self.lut).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs1000_units512_float32(self):
        act_idcs, w_idcs = self.create_inputs([1, 1000], [1000, 512])
        result = matmul_lut(act_idcs, w_idcs, self.lut_f32).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch32_inputs1000_units512_int32(self):
        act_idcs, w_idcs = self.create_inputs([32, 1000], [1000, 512])
        result = matmul_lut(act_idcs, w_idcs, self.lut).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch32_inputs1000_units512_float32(self):
        act_idcs, w_idcs = self.create_inputs([32, 1000], [1000, 512])
        result = matmul_lut(act_idcs, w_idcs, self.lut_f32).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch32_inputs1000_units1_int32(self):
        act_idcs, w_idcs = self.create_inputs([32, 1000], [1000, 1])
        result = matmul_lut(act_idcs, w_idcs, self.lut).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch32_inputs1000_units1_float32(self):
        act_idcs, w_idcs = self.create_inputs([32, 1000], [1000, 1])
        result = matmul_lut(act_idcs, w_idcs, self.lut_f32).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs1000_units1_int32(self):
        act_idcs, w_idcs = self.create_inputs([1, 1000], [1000, 1])
        result = matmul_lut(act_idcs, w_idcs, self.lut).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inputs1000_units1_float32(self):
        act_idcs, w_idcs = self.create_inputs([1, 1000], [1000, 1])
        result = matmul_lut(act_idcs, w_idcs, self.lut_f32).numpy()
        expected = self.compute_expected(act_idcs, w_idcs)
        self.assertTrue(np.array_equal(result, expected))



if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    # load custom op lib
    _lut_ops_module = tf.load_op_library(
            '/home/lhannink/code/git/tensorflow/bazel-bin/tensorflow/examples/lut_ops/lut_ops_op_lib.so')
    matmul_lut = _lut_ops_module.mat_mul_lut

    unittest.main()
