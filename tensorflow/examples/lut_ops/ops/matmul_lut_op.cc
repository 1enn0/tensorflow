#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tf = tensorflow;

REGISTER_OP("MatMulLut")
    .Input("activation_indices: InputIdxType")
    .Input("weight_indices: InputIdxType")
    .Input("lookup_table: LUTValueType")
    .Output("product: LUTValueType")
    .Attr("InputIdxType: {int32}")
    .Attr("LUTValueType: {int32, float}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(tf::shape_inference::MatMulShape);
