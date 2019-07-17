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
"""Benchmark for Matmul operator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import time

import numpy as np

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.dtypes import cast
from tensorflow import ConfigProto


def build_graph(device, mat_mul_algo, n, m, k, transpose_a, transpose_b, dtype_input,
                n_activations=None, n_weights=None, dtype_lut=None):
  """Build a graph containing a sequence of matmul operations.

  Args:
    device: String, the device to run on.
    mat_mul_algo: which algorithm to use. must be one of: ['opt' (standard tensorflow impl),
    'naive' (naive nested-for-loop impl), 'lut' (impl using lookup table)]
    n: tensor A's first dimension size.
    m: tensor A's second dimension size.
    k: tensor B's second dimension size.
    transpose_a: boolean value to show if tensor A is transposed.
    transpose_b: boolean value to show if tensor B is transposed.
    dtype_input: numpy data type of the input tensor.
    n_activations: number of discrete activation steps
    n_weights: number of discrete weight values
    dtype_lut: numpy data type of the LUT tensor

  Returns:
    A matmul operation to run()
  """
  with ops.device('%s' % device):
      if np.issubdtype(dtype_input, np.integer):
          d = np.int32
      else:
          d = dtype_input

      a = variables.VariableV1(
            random_ops.random_uniform(
                [n, m] if not transpose_a else [m, n],
                maxval=n_activations,
                dtype=d))
      b = variables.VariableV1(
            random_ops.random_uniform(
                [m, k] if not transpose_b else [k, m],
                maxval=n_weights,
                dtype=d))

      if np.issubdtype(dtype_input, np.integer):
          assert n_activations < np.iinfo(dtype_input).max
          assert n_weights < np.iinfo(dtype_input).max
          a = cast(a, dtype_input)
          b = cast(b, dtype_input)

      if mat_mul_algo == 'opt':
          z = math_ops.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
      elif mat_mul_algo == 'naive':
          z = gen_user_ops.mat_mul_naive(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
      elif mat_mul_algo == 'lut' and n_activations is not None and n_weights is not None and dtype_lut is not None:
          lut = variables.VariableV1(random_ops.random_uniform([n_activations, n_weights], maxval=10000, dtype=dtype_lut))
          z = gen_user_ops.mat_mul_lut(a, b, lut, transpose_a=transpose_a, transpose_b=transpose_b)
      else:
          raise ValueError(f'invalid argument: \'mat_mul_algo\' must be one of [\'opt\', \'naive\', \'lut\'] \
              but is: \'{mat_mul_algo}\'')

      return control_flow_ops.group(z)

class MatmulLUTBenchmark(test.Benchmark):
  """Benchmark mat_mul_lut!"""

  def run_graph(self, device, n, m, k, transpose_a, transpose_b, dtype_input,
                n_activations, n_weights, dtype_lut, num_iters):
    """Run the graph and print its execution time.

    Args:
      device: String, the device to run on.
      n: tensor A's first dimension size.
      m: tensor A's second dimension size.
      k: tensor B's second dimension size.
      transpose_a: boolean value to show if tensor A is transposed.
      transpose_b: boolean value to show if tensor B is transposed.
      dtype_input: numpy data type of the input tensor.
      n_activations: number of discrete activation steps
      n_weights: number of discrete weight values
      dtype_lut: numpy data type of the LUT tensor
      num_iters: number of iterations to run the benchmark.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      output = build_graph(device, 'lut', n, m, k, transpose_a, transpose_b, dtype_input, n_activations=n_activations,
                           n_weights=n_weights, dtype_lut=dtype_lut)
      with session_lib.Session(graph=graph, config=ConfigProto(device_count={'GPU': 0})) as session:
        variables.global_variables_initializer().run()
        for _ in range(500):
          session.run(output)
        start_time = time.time()
        for _ in range(num_iters):
          session.run(output)
        duration = (time.time() - start_time)
        num_items = n * m * k * 2
        throughput = num_items * num_iters / duration / 1e9
        print(f'{device} {dtype_input} input_info:{n}x{m}x{k} ta:{transpose_a} tb:{transpose_b}, lut_info: {dtype_lut} {n_activations}x{n_weights}, {num_iters}, {duration:.4f} s, {throughput:.4f} Gitems/s')

    name_template = ('mat_mul_lut_input_info_{dtype_input}_{inputinfo}_{dtype_lut}_lut_info_{lutinfo}')

    self.report_benchmark(
        name=name_template.format(
            dtype_input=str(dtype_input).replace(' ', ''),
            inputinfo=f'{n}x{m}x{k} ta:{transpose_a} tb:{transpose_b},',
            dtype_lut=str(dtype_lut).replace(' ', ''),
            lutinfo=f'{n_activations}x{n_weights}'),
        iters=num_iters,
        wall_time=duration)
    return duration

  def test_round(self, num_iters):
    input_sizes = [256, 512, 1024]
    transposes = [(False, False), (True, False), (False, True)]
    n_weightss = [500, 1000]
    n_activationss = [16, 32, 64, 128, 256]
    dtypes_input = [np.uint16, np.uint32]
    dtypes_lut = [np.int32, np.int64]
    configs = itertools.product(input_sizes, input_sizes, transposes, n_activationss, n_weightss, dtypes_input, dtypes_lut)
    for n, m, (transpose_a, transpose_b), n_activations, n_weights, dtype_input, dtype_lut in configs:
        self.run_graph('', n, m, n, transpose_a, transpose_b, dtype_input, n_activations, n_weights, dtype_lut, num_iters)

  def benchmark_matmul(self):
    self.test_round(num_iters=200)


class MatmulNaiveBenchmark(test.Benchmark):
  """Benchmark matmul!"""

  def run_graph(self, device, n, m, k, transpose_a, transpose_b, dtype, num_iters):
    """Run the graph and print its execution time.

    Args:
      device: String, the device to run on.
      n: tensor A's first dimension size.
      m: tensor A's second dimension size.
      k: tensor B's second dimension size.
      transpose_a: boolean value to show if tensor A is transposed.
      transpose_b: boolean value to show if tensor B is transposed.
      num_iters: number of iterations to run the benchmark.
      dtype: numpy data type of the input tensor.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      output = build_graph(device, 'naive', n, m, k, transpose_a, transpose_b, dtype)
      with session_lib.Session(graph=graph, config=ConfigProto(device_count={'GPU': 0})) as session:
        variables.global_variables_initializer().run()
        for _ in range(500):
          session.run(output)
        start_time = time.time()
        for _ in range(num_iters):
          session.run(output)
        duration = (time.time() - start_time)
        num_items = n * m * k * 2
        throughput = num_items * num_iters / duration / 1e9
        print('%s %s input_info:%s %d %.4fsec, %.4fGitems/s.' %
              (device, str(dtype), str(n) + 'x' + str(m) + 'x' + str(k) +
               ',ta:' + str(transpose_a) + '.tb:' + str(transpose_b), num_iters,
               duration, throughput))

    name_template = ('matmul_{device}_{dtype}_input_info_{inputinfo}')

    self.report_benchmark(
        name=name_template.format(
            device=device,
            dtype=str(dtype).replace(' ', ''),
            inputinfo=str(n) + 'x' + str(m) + 'x' + str(k) + ',ta:' +
            str(transpose_a) + ',tb:' + str(transpose_b)).replace(' ', ''),
        iters=num_iters,
        wall_time=duration)
    return duration

  def run_test_gpu(self, n, m, k, transpose_a, transpose_b, dtype, num_iters):
    self.run_graph(test.gpu_device_name(), n, m, k, transpose_a, transpose_b,
                   num_iters, dtype)

  def test_round(self, num_iters):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      for n, m, (transpose_a, transpose_b) in itertools.product(
          [512, 1024], [1, 8, 16, 128], [(False, False), (True, False),
                                         (False, True)]):
        k = n
        self.run_graph('', n, m, k, transpose_a, transpose_b, dtype, num_iters)

      for n, m, k, (transpose_a, transpose_b) in itertools.product(
          [200], [1, 8, 20], [10000], [(False, False), (True, False),
                                       (False, True)]):
        self.run_graph('', n, m, k, transpose_a, transpose_b, dtype, num_iters)

      for (n, m, k), (transpose_a, transpose_b) in itertools.product(
          [(200, 20, 20000), (1, 10000, 200)], [(False, False), (True, False),
                                                (False, True)]):
        self.run_graph('', n, m, k, transpose_a, transpose_b, dtype, num_iters)

  def benchmark_matmul(self):
    self.test_round(num_iters=200)


if __name__ == '__main__':
  test.main()
