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

#include <fstream>
#include <vector>
#include <chrono>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/examples/parameter_estimation/estimate_params_from_wav.h"

namespace tf = tensorflow;

static const tf::string input_name_pre{"wav_data"};
static const tf::string output_name_pre{"fingerprint_output"};
static const tf::string input_name_model{"reshape_input"};
static const tf::string output_name_model{"model_out0"};

namespace {
//-----------------------------------------------------------------------------
// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
tf::Status loadGraph(std::unique_ptr<tf::Session>* session,
                     const tf::string& graph_file_name) {
  tf::GraphDef graph_def;
  tf::Status status;

  status = ReadBinaryProto(tf::Env::Default(), graph_file_name, &graph_def);
  if (!status.ok()) {
    return tf::errors::NotFound("Failed to load compute graph at '",
                                graph_file_name, "'");
  }

  session->reset(tf::NewSession(tf::SessionOptions()));
  status = (*session)->Create(graph_def);
  if (!status.ok()) {
    return status;
  }
  LOG(INFO) << "successfully loaded graph at '" << graph_file_name << "'\n";

  return tf::Status::OK();
}
}  // namespace

//-----------------------------------------------------------------------------
tf::Status wavToLogMelSpectrogram(std::unique_ptr<tf::Session>* session,
                                  const tf::string& input_wav_path,
                                  const tf::string& graph,
                                  tf::Tensor& spectrogram_out) {
  tf::Status status;
  status = loadGraph(session, graph);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return status;
  }

  tf::string wav_string;
  status =
      tf::ReadFileToString(tf::Env::Default(), input_wav_path, &wav_string);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return status;
  }

  auto wav_tensor = tf::Tensor(wav_string);

  std::vector<tf::Tensor> outputs;
  status = (*session)->Run({{input_name_pre, wav_tensor}}, {output_name_pre},
                           {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Running model failed: " << status;
    return status;
  }

  if (!outputs.empty()) spectrogram_out = outputs[0];

  return tf::Status::OK();
}

//-----------------------------------------------------------------------------
tf::Status runInference(std::unique_ptr<tf::Session>* session,
                        const tf::string& graph,
                        const tf::Tensor& spectrogram_in,
                        tf::Tensor& params_out) {

  tf::Status status;
  status = loadGraph(session, graph);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return status;
  }

  std::vector<tf::Tensor> outputs;

  auto start = std::chrono::high_resolution_clock::now();
  status = (*session)->Run({{input_name_model, spectrogram_in}},
                           {output_name_model}, {}, &outputs);
  auto end = std::chrono::high_resolution_clock::now();

  if (!status.ok()) {
    LOG(ERROR) << "Running model failed: " << status;
    return status;
  }

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  LOG(INFO) << "execution of graph took " << duration.count() << " ns.\n";


  if(!outputs.empty()) params_out = outputs[0];

  return tf::Status::OK();
}