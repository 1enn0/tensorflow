
#include <array>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"


//-----------------------------------------------------------------------------
namespace custom_ops {

using CPUDevice = Eigen::ThreadPoolDevice;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;
using tensorflow::OpKernel;
using tensorflow::OpKernelContext;
using tensorflow::OpKernelConstruction;

//-----------------------------------------------------------------------------
template <typename T>
class MatMulNaiveOp : public OpKernel {
//-----------------------------------------------------------------------------
  public:
    explicit MatMulNaiveOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    }
    
    void Compute(OpKernelContext* ctx) override 
    {
      // get input tensors
      const Tensor& a = ctx->input(0);
      const Tensor& b = ctx->input(1);

      // check that dimensions of the two input matrices are valid
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsMatrix(a.shape()),
          tensorflow::errors::InvalidArgument("In[0] is not a matrix. Instead it has shape ",
                                  a.shape().DebugString()));
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsMatrix(b.shape()),
          tensorflow::errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                  b.shape().DebugString()));

      int a_dim_product = transpose_a_ ? 0 : 1;
      int a_dim_remaining = 1 - a_dim_product;
      int b_dim_product = transpose_b_ ? 1 : 0;
      int b_dim_remaining = 1 - b_dim_product;

      OP_REQUIRES(
          ctx, a.dim_size(a_dim_product) == b.dim_size(b_dim_product),
          tensorflow::errors::InvalidArgument(
            "Matrix size-incompatible: In[1]: ", a.shape().DebugString(),
            ", In[2]: ", b.shape().DebugString()));

      TensorShape out_shape(
          {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
      Tensor* out_tensor {nullptr};
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out_tensor));

      if (out_tensor->NumElements() == 0) {
        // If a has shape [0, x] or b has shape [x, 0], the output shape
        // is a 0-element matrix, so there is nothing to do.
        return;
      }

      /* if (a.NumElements() == 0 || b.NumElements() == 0) */
      /* { */
      /*   // if a has shape [x, 0] and b has shape [0, y] */
      /*   // (where x and y are non-zero), the output shape is */
      /*   // [x, y] and we just fill it with zeros */
      /*   tensorflow::functor::SetZeroFunctor<CPUDevice, LUTValueType> f; */
      /*   f(ctx->eigen_device<CPUDevice>(), out_tensor->flat<LUTValueType>()); */
      /* } */

      auto a_values = a.matrix<T>();
      auto b_values = b.matrix<T>(); 
      auto output_values = out_tensor->matrix<T>();

      for (int i {0}; i < a.dim_size(a_dim_remaining); ++i)
      {
        for (int j {0}; j < b.dim_size(b_dim_remaining); ++j)
        {
          output_values(i, j) = 0;
          for (int k {0}; k < a.dim_size(a_dim_product); ++k)
          {
            auto a_ = transpose_a_ ? a_values(k, i) : a_values(i, k);
            auto b_ = transpose_b_ ? b_values(j, k) : b_values(k, j);
            output_values(i, j) += a_ * b_;
          }
        }
      }
    }

//-----------------------------------------------------------------------------
  private:
    bool transpose_a_ {false};
    bool transpose_b_ {false};
};
//-----------------------------------------------------------------------------

} // custom_ops


//-----------------------------------------------------------------------------
REGISTER_OP("MatMulNaive")
  .Input("a: T")
  .Input("b: T")
  .Output("product: T")
  .Attr("transpose_a: bool = false")
  .Attr("transpose_b: bool = false")
  .Attr("T: {float, double}")
  .SetShapeFn(tensorflow::shape_inference::MatMulShape);


//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("MatMulNaive")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("T"),
    custom_ops::MatMulNaiveOp<float>);

//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("MatMulNaive")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("T"),
    custom_ops::MatMulNaiveOp<double>);
