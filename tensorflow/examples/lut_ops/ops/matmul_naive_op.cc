#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tf = tensorflow;

REGISTER_OP("MatMulNaiveV2")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: {int32, int64, float, double}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(tf::shape_inference::MatMulShape);
