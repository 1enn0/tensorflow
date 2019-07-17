
#pragma once

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

using SessionPtr = std::unique_ptr<tf::Session>;

tf::Status wavToLogMelSpectrogram(SessionPtr& session,
                                  const tf::string& input_wav_path,
                                  const tf::string& graph,
                                  tf::Tensor& spectrogram_out);

tf::Status runInference (SessionPtr& session,
                         const tf::string& graph,
                         const tf::Tensor& spectrogram_in,
                         tf::Tensor& params_out);
