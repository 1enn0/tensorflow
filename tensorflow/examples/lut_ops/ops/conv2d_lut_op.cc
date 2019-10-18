#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tf = tensorflow;

REGISTER_OP("Conv2dLut")
    .Input("input: InputIdxType")
    .Input("filter: InputIdxType")
    .Input("lookup_table: LUTValueType")
    .Output("output: LUTValueType")
    .Attr("InputIdxType: {int32}")
    .Attr("LUTValueType: {int32, float}")
    .Attr("strides: list(int)")
    .Attr(tf::GetPaddingAttrString())
    .Attr(tf::GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(tf::shape_inference::Conv2DShape);
