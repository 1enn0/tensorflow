#include <array>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/kernels/matmul_op.h"

namespace tf = tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;

namespace tensorflow {
namespace functor {
template <typename T>
struct LaunchMatMulCPU;
}}

//-----------------------------------------------------------------------------
template <typename T>
class MyCompositeOp : public tf::OpKernel {
//-----------------------------------------------------------------------------
  using Tensor = tf::Tensor;
  using TensorShape = tf::TensorShape;
  using TensorShapeUtils = tf::TensorShapeUtils;

  public:

    explicit MyCompositeOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    }
    
    void Compute(tf::OpKernelContext* ctx) override 
    {
      // get input tensors
      const Tensor& a = ctx->input(0);
      const Tensor& b = ctx->input(1);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0].first = transpose_a_ ? 0 : 1;
      dim_pair[0].second = transpose_b_ ? 1 : 0;
      int a_dim_remaining = 1 - dim_pair[0].first;
      int b_dim_remaining = 1 - dim_pair[0].second;
      TensorShape out_shape(
          {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
      

      Tensor* output_tensor {nullptr};
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output_tensor));

      tensorflow::functor::MatMul(ctx->eigen_device<CPUDevice>(), 
          output_tensor->matrix<T>(), a.matrix<T>(), b.matrix<T>(), dim_pair);
      /* tensorflow::functor::LaunchMatMulCPU<T>::launch( */
      /*     ctx, a, b, dim_pair, nullptr, false, output_tensor); */

      /* auto result = tf::ops::MatMul(a, b, tf::ops::MatMul::Attrs().TransposeA(transpose_a_).TransposeB(transpose_b_)); */


      /* OP_REQUIRES_OK(ctx, ctx->set_output(result.tensor_data(), result)); */
    }

//-----------------------------------------------------------------------------
  private:
    bool transpose_a_ {false};
    bool transpose_b_ {false};
};
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
REGISTER_OP("MyComposite")
  .Input("a: T")
  .Input("b: T")
  .Output("product: LUTValueType")
  .Attr("transpose_a: bool = false")
  .Attr("transpose_b: bool = false")
  .Attr("T: {float, double} = DT_FLOAT")
  .SetShapeFn(tf::shape_inference::MatMulShape);

//-----------------------------------------------------------------------------
#define REGISTER_MY_COMPOSITE_KERNEL_BUILDER(T) \
  REGISTER_KERNEL_BUILDER(                      \
      Name("MyComposite")                       \
      .Device(tf::DEVICE_CPU)                   \
      .TypeConstraint<T>("T"),                  \
      MyCompositeOp<T>) 

//-----------------------------------------------------------------------------
REGISTER_MY_COMPOSITE_KERNEL_BUILDER(float);
REGISTER_MY_COMPOSITE_KERNEL_BUILDER(double);
