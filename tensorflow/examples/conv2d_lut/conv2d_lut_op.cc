/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tf = tensorflow;

REGISTER_OP("Conv2DLUT")
    .Input("input: InputIdxType")
    .Input("filter: InputIdxType")
    .Input("lookup_table: LUTValueType")
    .Output("output: LUTValueType")
    .Attr("InputIdxType: {int32}")
    .Attr("LUTValueType: {int32, float}")
    .Attr("strides: list(int)")
    .Attr(tf::GetPaddingAttrStringWithExplicit())
    .Attr(tf::GetExplicitPaddingsAttrString())
    .Attr(tf::GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(tf::shape_inference::Conv2DShapeWithExplicitPadding);
