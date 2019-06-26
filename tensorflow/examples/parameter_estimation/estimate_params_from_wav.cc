/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/examples/parameter_estimation/estimate_params_from_wav.h"

namespace {

namespace tf = tensorflow;

using tf::Session;
using tf::Status;
using tf::string;
using tf::Tensor;

using FloatMilliseconds =
        std::chrono::duration<float, std::chrono::milliseconds::period>;

//TODO: refacto standard graph in/out node names
static const string input_name_pre{"wav_data"};
static const string output_name_pre{"fingerprint_output"};
static const string input_name_model{"reshape_input"};
static const string output_name_model{"model_out0"};

//-----------------------------------------------------------------------------
// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status loadGraph(SessionPtr& session, const string& graph_file_path) {
  tf::GraphDef graph_def;
  Status status;
  status = ReadBinaryProto(tf::Env::Default(), graph_file_path, &graph_def);

  if (!status.ok()) {
    return tf::errors::NotFound("failed to load compute graph at '",
                                graph_file_path, "'");
  }

  session.reset(tf::NewSession(tf::SessionOptions()));
  status = session->Create(graph_def);
  if (!status.ok()) {
    return status;
  }
  LOG(INFO) << "loaded graph at '" << graph_file_path <<  "'\n";

  return Status::OK();
}


//-----------------------------------------------------------------------------
template <typename T>
void printStats(const std::vector<T>& vec)
{
  auto max_elt = std::max_element(vec.begin(), vec.end());
  auto min_elt = std::min_element(vec.begin(), vec.end());
  T sum {};
  for (auto elt : vec)
    sum += elt;
  T mean = sum / T(vec.size());
  LOG(INFO) << "max: " << *max_elt << ", min: " << *min_elt << ", mean: " << mean;
}



}  // namespace

//-----------------------------------------------------------------------------
Status wavToLogMelSpectrogram(SessionPtr& session, const string& input_wav_path,
                              const string& graph, Tensor& spectrogram_out) {
  Status status;
  status = loadGraph(session, graph);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return status;
  }

  string wav_string;
  status =
      tf::ReadFileToString(tf::Env::Default(), input_wav_path, &wav_string);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return status;
  }

  auto wav_tensor = Tensor(wav_string);

  std::vector<Tensor> outputs;
  status = session->Run({{input_name_pre, wav_tensor}}, {output_name_pre},
                           {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Running model failed: " << status;
    return status;
  }

  if (!outputs.empty()) spectrogram_out = outputs[0];

  return Status::OK();
}

//-----------------------------------------------------------------------------
Status runInference(SessionPtr& session, const string& graph,
                    const Tensor& spectrogram_in, Tensor& params_out) {
  Status status;
  status = loadGraph(session, graph);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return status;
  }

  std::vector<Tensor> outputs;
  size_t n_iterations = 100;
  std::vector<float> timings{};

  for (size_t i{0}; i < n_iterations; ++i) {
    outputs.clear();
    auto start = std::chrono::high_resolution_clock::now();
    status = session->Run({{input_name_model, spectrogram_in}},
                             {output_name_model}, {}, &outputs);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = FloatMilliseconds(end - start);
    timings.push_back(duration.count());
    LOG(INFO) << "execution of graph took " << duration.count() << " ms.";
  }

  printStats(timings);

  if (!status.ok()) {
    LOG(ERROR) << "Running model failed: " << status;
    return status;
  }

  if (!outputs.empty()) params_out = outputs[0];

  return Status::OK();
}
