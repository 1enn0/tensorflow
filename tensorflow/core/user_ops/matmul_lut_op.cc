
#include <array>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
/* #include "tensorflow/core/kernels/fill_functor.h" */

namespace tf = tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;

//-----------------------------------------------------------------------------
template <typename InputIdxType, typename LUTValueType>
class MatMulLUTOp : public tf::OpKernel {
//-----------------------------------------------------------------------------
  using Tensor = tf::Tensor;
  using TensorShape = tf::TensorShape;
  using TensorShapeUtils = tf::TensorShapeUtils;

  public:

    explicit MatMulLUTOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    }
    
    void Compute(tf::OpKernelContext* ctx) override 
    {
      // get input tensors
      const Tensor& a = ctx->input(0);
      const Tensor& b = ctx->input(1);
      const Tensor& t = ctx->input(2);

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

      auto a_values = a.matrix<InputIdxType>();
      auto b_values = b.matrix<InputIdxType>(); 
      auto lut = t.matrix<LUTValueType>();
      auto output_values = out_tensor->matrix<LUTValueType>();

      for (int i {0}; i < a.dim_size(a_dim_remaining); ++i)
      {
        for (int j {0}; j < b.dim_size(b_dim_remaining); ++j)
        {
          output_values(i, j) = 0;
          for (int k {0}; k < a.dim_size(a_dim_product); ++k)
          {
            auto a_idx = transpose_a_ ? a_values(k, i) : a_values(i, k);
            auto b_idx = transpose_b_ ? b_values(j, k) : b_values(k, j);
            output_values(i, j) += lut(a_idx, b_idx);
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

//-----------------------------------------------------------------------------
REGISTER_OP("MatMulLUT")
  .Input("activation_indices: InputIdxType")
  .Input("weight_indices: InputIdxType")
  .Input("lookup_table: LUTValueType")
  .Output("product: LUTValueType")
  .Attr("transpose_a: bool = false")
  .Attr("transpose_b: bool = false")
  .Attr("InputIdxType: {uint8, uint16, uint32} = DT_UINT16")
  .Attr("LUTValueType: {int32, int64} = DT_INT32")
  .SetShapeFn(tf::shape_inference::MatMulShape);

//-----------------------------------------------------------------------------
#define REGISTER_MATMUL_LUT_KERNEL_BUILDER(T,U) \
  REGISTER_KERNEL_BUILDER(                      \
      Name("MatMulLUT")                         \
      .Device(tf::DEVICE_CPU)                   \
      .TypeConstraint<T>("InputIdxType")        \
      .TypeConstraint<U>("LUTValueType"),       \
      MatMulLUTOp<T, U>) 

//-----------------------------------------------------------------------------
REGISTER_MATMUL_LUT_KERNEL_BUILDER(tf::uint8, tf::int32);
REGISTER_MATMUL_LUT_KERNEL_BUILDER(tf::uint16, tf::int32);
REGISTER_MATMUL_LUT_KERNEL_BUILDER(tf::uint32, tf::int32);
REGISTER_MATMUL_LUT_KERNEL_BUILDER(tf::uint8, tf::int64);
REGISTER_MATMUL_LUT_KERNEL_BUILDER(tf::uint16, tf::int64);
REGISTER_MATMUL_LUT_KERNEL_BUILDER(tf::uint32, tf::int64);
