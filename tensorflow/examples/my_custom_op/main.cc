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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"

int main(int argc, char* argv[]) {
  using namespace tensorflow;
  using namespace tensorflow::ops;

  Scope root = Scope::NewRootScope();

  auto ops = OpRegistry::Global()->DebugString(false);
  const string opsFilePath {"/tmp/ops.pbtxt"};
  WriteStringToFile(Env::Default(), opsFilePath, ops);

  auto a = Const(root.WithOpName("a"), {uint8(0), uint8(1), uint8(2), uint8(3)});
  auto b = Const(root.WithOpName("b"), {uint8(0), uint8(1), uint8(2), uint8(3)});
  auto res = DotLUT(root.WithOpName("dotLUT"), a, b);

  std::vector<Tensor> outputs {};
  ClientSession session {root};
  TF_CHECK_OK (session.Run({res}, &outputs));
  LOG(INFO) << "outputs[0].shape() = " <<  outputs[0].shape() << "\n" << outputs[0].vec<float>();

  GraphDef graph_def;
  root.ToGraphDef(&graph_def);
  const string graph_file_path {"/tmp/graph.pbtxt"};
  WriteTextProto(Env::Default(), graph_file_path, graph_def);

  return 0;
}
