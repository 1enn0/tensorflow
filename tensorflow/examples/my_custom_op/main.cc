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

namespace {
using namespace tensorflow;
using namespace tensorflow::ops;

//-----------------------------------------------------------------------------
void lookupUsingGather()
{
  Scope root = Scope::NewRootScope();

  // approach 2: using const tensor
  /* auto lut = Const(root.WithOpName("lut"), {100.f, 101.f, 102.f, 103.f}); */
  auto lut = Variable(root, {4}, DT_FLOAT);
  auto assign_lut = Assign(root, lut, RandomUniform(root, {4}, DT_FLOAT));
  auto keys = Input({3, 2, 1, 0, 0, 1, 1, 1, 1, 2, 3, 2, 1, 0, 0, 1});
  auto res = Sum(root, Gather(root, lut, keys), 0);

  ClientSession session {root};

  std::vector<Tensor> outputs {};
  TF_CHECK_OK (session.Run({assign_lut}, nullptr));
  TF_CHECK_OK (session.Run({res}, &outputs));
  LOG(INFO) << "lookupUsingGather(): outputs[0].shape() = " <<  outputs[0].shape() << "\n" << outputs[0].flat<float>();
}

//-----------------------------------------------------------------------------
void lookupUsingHashTable()
{
  Scope root = Scope::NewRootScope();

  // approach 1: using hash table
  auto lut = MutableHashTable(root, DT_INT32, DT_FLOAT, MutableHashTable::SharedName("lut"));
  auto keys = Input({0, 1, 2, 3});
  auto values = Input({100.f, 101.f, 102.f, 103.f});
  auto lut_insert = LookupTableInsert(root, lut, keys, values);

  auto find_keys = Input({3, 2, 1, 0});
  auto res = Sum(root, LookupTableFind(root, lut, find_keys, -1.f), 0);

  ClientSession session {root};

  // initialize table by running insert op
  TF_CHECK_OK (session.Run({}, {}, {lut_insert}, nullptr));

  // fetch outputs
  std::vector<Tensor> outputs {};
  TF_CHECK_OK (session.Run({res}, &outputs));
  LOG(INFO) << "lookupUsingHashTable(): outputs[0].shape() = " <<  outputs[0].shape() << "\n" << outputs[0].flat<float>();
}

//-----------------------------------------------------------------------------
void lookupUsingMyCustomOp()
{
  Scope root = Scope::NewRootScope();

  auto a = Const(root.WithOpName("a"), {uint8(0), uint8(1), uint8(2), uint8(3)});
  auto b = Const(root.WithOpName("b"), {uint8(0), uint8(1), uint8(2), uint8(3)});
  auto res = DotProductLut(root.WithOpName("dotLUT"), a, b);
  auto lut = Variable(root, {4, 4}, DT_FLOAT);
  auto assign_lut = Assign(root.WithOpName("assign_lut"), lut, RandomUniform(root, {4, 4}, DT_FLOAT));

  ClientSession session {root};

  // initialize lut variable by running assign op
  TF_CHECK_OK (session.Run({assign_lut}, nullptr));

  // fetch outputs
  std::vector<Tensor> outputs {};
  TF_CHECK_OK (session.Run({res}, &outputs));
  LOG(INFO) << "lookupUsingMyCustomOp(): outputs[0].shape() = " <<  outputs[0].shape() << "\n" << outputs[0].flat<float>();
}

//-----------------------------------------------------------------------------
void writeOpsToFile () 
{
  auto ops = OpRegistry::Global()->DebugString(false);
  const string opsFilePath {"/tmp/ops.pbtxt"};
  WriteStringToFile(Env::Default(), opsFilePath, ops);
}

//-----------------------------------------------------------------------------
void writeGraphToTextFile (const Scope& root)
{
  GraphDef graph_def;
  root.ToGraphDef(&graph_def);
  const string graph_file_path {"/tmp/graph.pbtxt"};
  WriteTextProto(Env::Default(), graph_file_path, graph_def);
}

//-----------------------------------------------------------------------------
} // anonymous

//-----------------------------------------------------------------------------
int main(int argc, char* argv[]) 
{
  lookupUsingGather();
  lookupUsingHashTable();
  lookupUsingMyCustomOp();

  return 0;
}

