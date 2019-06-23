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

#include <memory>

#include "tensorflow/examples/parameter_estimation/estimate_params_from_wav.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tf = tensorflow;

int main(int argc, char* argv[]) {
  tf::string input_wav_path{};
  tf::string preprocessing_graph{};
  tf::string model_graph{};

  std::vector<tf::Flag> flag_list = {
      tf::Flag("input_wav_path", &input_wav_path, "audio file to load"),
      tf::Flag(
          "preprocessing_graph", &preprocessing_graph,
          "preprocessing graph to transform wave into log-mel spectrograms"),
      tf::Flag("model_graph", &model_graph, "model to execute"),
  };

  tf::string usage = tf::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tf::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tf::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  //   auto session =
  //   std::make_unique<tf::Session>(tf::NewSession(tf::SessionOptions()));
  std::unique_ptr<tf::Session> session;

  tf::Tensor spectrogram;
  tf::Status status;
  status = wavToLogMelSpectrogram(&session, input_wav_path,
                                  preprocessing_graph, spectrogram);
  
  if(!status.ok())
  {
      LOG(ERROR) << "wavToLogMelSpectrogram() failed.\n";
      return -1;
  }
  LOG(INFO) << "spectrogram shape: " << spectrogram.shape();

  tf::Tensor params;
  status = runInference(&session, model_graph, spectrogram, params);
  if(!status.ok())
  {
      LOG(ERROR) << "running model failed.\n";
      return -1;
  }
  LOG(INFO) << "params shape: " << params.shape() << "\nparams: " << (params).tensor<float, 2>();


  return 0;
}