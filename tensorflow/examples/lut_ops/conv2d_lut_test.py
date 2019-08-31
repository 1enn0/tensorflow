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

"""Test for conv2dlut op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os.path

import tensorflow as tf

import numpy as np

conv2dlut = None

class Conv2DLUTTest(unittest.TestCase):

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

        self.padding_value = self.activations[int(self.n_activations / 2)]

    def create_inputs(self, input_shape, kernel_shape):
        """Create input arrays.

        Args:
            input_shape: [batches, in_height, in_width, in_channels]
            kernel_shape: [kernel_height, kernel_width,
                kernel_channels, kernel_filters]

        Returns:
            the two arrays inputs, kernels
        """
        assert input_shape[-1] == kernel_shape[2], "kernel_channels has to be equal to in_channels"

        inputs = np.random.randint(
            0, self.n_activations, size=input_shape).astype(np.int32)
        kernels = np.random.randint(
            0, self.n_weights, size=kernel_shape).astype(np.int32)

        return inputs, kernels

    def compute_expected(self, input_idcs, kernel_idcs, strides, padding):
        """Compute expected output with regular conv2d function.

        For this, we look up the values corresponding
        to the input index arrays and cast them
        to float32 (as conv2d does not support int32) and
        cast the result back to int32.
        """
        input_values = self.activations[input_idcs].astype(np.float32)
        kernel_values = self.weights[kernel_idcs].astype(np.float32)

        if padding == 'SAME':
            # we have to manually pad the inputs and use 'VALID'
            # padding here, since we do not want to pad with zeroes
            # we use padding length [top, bottom, left, right] = 1 since
            # we only use [3, 3] kernels here in these tests
            input_shape_padded = [input_idcs.shape[0], input_idcs.shape[1] + 2, input_idcs.shape[2] + 2, input_idcs.shape[3]]
            input_values_padded = np.full(input_shape_padded, self.padding_value, dtype=np.float32)
            input_values_padded[:, 1:-1, 1:-1, :] = input_values
            input_values = input_values_padded
            padding = 'VALID'

        expected = tf.nn.conv2d(input_values, kernel_values, strides, padding)
        return expected.numpy().astype(np.int32)

    def test_batch1_inch1_kch1_pad_valid_nostride(self):
        """ test names have format:
        test_batch[number of batches]_inch[number of input channels]_kch[number of kernels]_pad_[padding mode]
        """
        input_idcs, kernel_idcs = self.create_inputs([1, 7, 7, 1], [3, 3, 1, 1])
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch2_kch1_pad_valid_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 7, 7, 2], [3, 3, 2, 1])
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch1_kch2_pad_valid_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 7, 7, 1], [3, 3, 1, 2])
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch2_kch2_pad_valid_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 7, 7, 2], [3, 3, 2, 2])
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch1_kch1_pad_valid_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 9, 9, 1], [3, 3, 1, 1])
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch2_kch1_pad_valid_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 9, 9, 2], [3, 3, 2, 1])
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch1_kch2_pad_valid_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 9, 9, 1], [3, 3, 1, 2])
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch2_kch2_pad_valid_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 9, 9, 2], [3, 3, 2, 2])
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    # SAME padding

    def test_batch1_inch1_kch1_pad_same_nostride(self):
        """ test names have format:
        test_batch[number of batches]_inch[number of input channels]_kch[number of kernels]_pad_[padding mode]
        """
        input_idcs, kernel_idcs = self.create_inputs([1, 7, 7, 1], [3, 3, 1, 1])
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch2_kch1_pad_same_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 7, 7, 2], [3, 3, 2, 1])
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch1_kch2_pad_same_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 7, 7, 1], [3, 3, 1, 2])
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch2_kch2_pad_same_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 7, 7, 2], [3, 3, 2, 2])
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch1_kch1_pad_same_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 9, 9, 1], [3, 3, 1, 1])
        strides = [1, 2, 2, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch2_kch1_pad_same_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 9, 9, 2], [3, 3, 2, 1])
        strides = [1, 2, 2, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch1_kch2_pad_same_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 9, 9, 1], [3, 3, 1, 2])
        strides = [1, 2, 2, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch1_inch2_kch2_pad_same_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([1, 9, 9, 2], [3, 3, 2, 2])
        strides = [1, 2, 2, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))


        # multiple batches

    def test_batch2_inch1_kch1_pad_valid_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 7, 7, 1], [3, 3, 1, 1])
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch2_kch1_pad_valid_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 7, 7, 2], [3, 3, 2, 1])
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch1_kch2_pad_valid_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 7, 7, 1], [3, 3, 1, 2])
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch2_kch2_pad_valid_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 7, 7, 2], [3, 3, 2, 2])
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch1_kch1_pad_valid_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 9, 9, 1], [3, 3, 1, 1])
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch2_kch1_pad_valid_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 9, 9, 2], [3, 3, 2, 1])
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch1_kch2_pad_valid_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 9, 9, 1], [3, 3, 1, 2])
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch2_kch2_pad_valid_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 9, 9, 2], [3, 3, 2, 2])
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    # SAME padding

    def test_batch2_inch1_kch1_pad_same_nostride(self):
        """ test names have format:
        test_batch[number of batches]_inch[number of input channels]_kch[number of kernels]_pad_[padding mode]
        """
        input_idcs, kernel_idcs = self.create_inputs([2, 7, 7, 1], [3, 3, 1, 1])
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch2_kch1_pad_same_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 7, 7, 2], [3, 3, 2, 1])
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch1_kch2_pad_same_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 7, 7, 1], [3, 3, 1, 2])
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch2_kch2_pad_same_nostride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 7, 7, 2], [3, 3, 2, 2])
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch1_kch1_pad_same_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 9, 9, 1], [3, 3, 1, 1])
        strides = [1, 2, 2, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch2_kch1_pad_same_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 9, 9, 2], [3, 3, 2, 1])
        strides = [1, 2, 2, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch1_kch2_pad_same_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 9, 9, 1], [3, 3, 1, 2])
        strides = [1, 2, 2, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))

    def test_batch2_inch2_kch2_pad_same_withstride(self):
        input_idcs, kernel_idcs = self.create_inputs([2, 9, 9, 2], [3, 3, 2, 2])
        strides = [1, 2, 2, 1]
        padding = 'SAME'
        result = conv2dlut(input_idcs, kernel_idcs, self.lut, strides, padding).numpy()
        expected = self.compute_expected(input_idcs, kernel_idcs, strides, padding)
        self.assertTrue(np.array_equal(result, expected))



if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    # load custom op lib
    _conv2d_lut_module = tf.load_op_library(
            '/home/lhannink/code/git/tensorflow/bazel-bin/tensorflow/examples/conv2d_lut/conv2d_lut_op_kernel.so')
    conv2dlut = _conv2d_lut_module.conv2dlut

    unittest.main()
