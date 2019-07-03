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
void lookupUsingGather()
{
  Scope root = Scope::NewRootScope();

  auto lut = Variable(root, {4}, DT_FLOAT);
  auto lut_values = Input({42.f, 5.f, .5f, 2.f});
  auto assign_lut = Assign(root, lut, lut_values);
  auto keys = Input({3, 2, 1, 0, 0, 1, 1});
  auto res = Sum(root, Gather(root, lut, keys), 0);

  ClientSession session {root};
  std::vector<Tensor> outputs {};
  TF_CHECK_OK (session.Run({assign_lut}, nullptr));
  TF_CHECK_OK (session.Run({res}, &outputs));
  LOG(INFO) << "lookupUsingGather(): outputs[0].shape() = " <<  outputs[0].shape() << "\n" << outputs[0].flat<float>();
}

//-----------------------------------------------------------------------------
void lookupUsingGatherNd()
{
  Scope root = Scope::NewRootScope();

  auto lut2d = Variable(root, {2, 2}, DT_FLOAT);
  auto lut2dValues = Input({
    {42.f, 5.f}, {.5f, 2.f},
  });
  auto assign_lut2d = Assign (root, lut2d, lut2dValues);
  auto keys2d = Input({
      {1, 1}, {1, 0}, {0, 1}, {0, 0}, {0, 0}, {0, 1}, {0, 1}
  });
  auto res2d = Sum(root, GatherNd(root, lut2d, keys2d), 0);

  ClientSession session {root};
  std::vector<Tensor> outputs {};
  TF_CHECK_OK (session.Run({assign_lut2d}, nullptr));
  TF_CHECK_OK (session.Run({res2d}, &outputs));
  LOG(INFO) << "lookupUsingGatherNd(): outputs[0].shape() = " <<  outputs[0].shape() << "\n" << outputs[0].flat<float>();
}

//-----------------------------------------------------------------------------
void lookupUsingHashTable()
{
  Scope root = Scope::NewRootScope();

  // approach 1: using hash table
  auto lut = MutableHashTable(root, DT_INT32, DT_FLOAT, MutableHashTable::SharedName("lut"));
  auto keys = Input({0, 1, 2, 3});
  auto values = Input({42.f, 5.f, .5f, 2.f});
  auto lut_insert = LookupTableInsert(root, lut, keys, values);

  auto find_keys = Input({3, 2, 1, 0, 0, 1, 1});
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

  auto lut = Variable(root, {2, 2}, DT_FLOAT);
  auto lutValues = Input({
    {42.f, 5.f}, {.5f, 2.f},
  });
  auto assign_lut = Assign(root.WithOpName("assign_lut"), lut, lutValues);

  auto a = Const(root.WithOpName("a"), {{1u, 1u, 0u, 0u, 0u, 0u, 0u}});
  /* auto b = Const(root.WithOpName("b"), {{1u}, {0u}, {1u}, {0u}, {0u}, {1u}, {1u}}); */
  /* auto res = MatMulLut(root, lut, a, b); */
  auto b = Const(root.WithOpName("b"), {{1u, 0u, 1u, 0u, 0u, 1u, 1u}});
  auto res = MatMulLut(root, lut, a, b, MatMulLut::TransposeB(true));


  ClientSession session (root);
  /* writeGraphToTextFile(root); */

  // initialize lut variable by running assign op
  TF_CHECK_OK (session.Run({assign_lut}, nullptr));

  // fetch outputs
  std::vector<Tensor> outputs {};
  TF_CHECK_OK (session.Run({res}, &outputs));
  LOG(INFO) << "lookupUsingMyCustomOp(): outputs[0].shape() = " <<  outputs[0].shape() << "\n" << outputs[0].flat<float>();


  GraphDef graphDef;
  TF_CHECK_OK(root.ToGraphDef(&graphDef));
  WriteBinaryProto(Env::Default(), "/tmp/graph.pb", graphDef);
}


//-----------------------------------------------------------------------------
} // anonymous

//-----------------------------------------------------------------------------
int main(int argc, char* argv[]) 
{
  /* lookupUsingGather(); */
  /* lookupUsingGatherNd(); */
  /* lookupUsingHashTable(); */
  lookupUsingMyCustomOp();

  return 0;
}

