
#include <array>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"


//-----------------------------------------------------------------------------
namespace custom_ops {

using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;
using tensorflow::OpKernel;
using tensorflow::OpKernelContext;
using tensorflow::OpKernelConstruction;
using tensorflow::uint8;
using tensorflow::uint16;
using tensorflow::uint32;
using tensorflow::int64;

//-----------------------------------------------------------------------------
template <typename T>
class MatMulLutOp : public OpKernel {
//-----------------------------------------------------------------------------
  public:
    explicit MatMulLutOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    }
    
    void Compute(OpKernelContext* ctx) override 
    {
      // get input tensors
      const Tensor& t = ctx->input(0);
      const Tensor& a = ctx->input(1);
      const Tensor& b = ctx->input(2);

      // check that dimensions of the two input matrices are valid
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsMatrix(a.shape()),
          tensorflow::errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                  a.shape().DebugString()));
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsMatrix(b.shape()),
          tensorflow::errors::InvalidArgument("In[2] is not a matrix. Instead it has shape ",
                                  b.shape().DebugString()));

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0].first = transpose_a_ ? 0 : 1;
      dim_pair[0].second = transpose_b_ ? 1 : 0;

      OP_REQUIRES(
          ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
          tensorflow::errors::InvalidArgument(
            "Matrix size-incompatible: In[1]: ", a.shape().DebugString(),
            ", In[2]: ", b.shape().DebugString()));

      int a_dim_remaining = 1 - dim_pair[0].first;
      int b_dim_remaining = 1 - dim_pair[0].second;
      TensorShape out_shape(
          {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
      Tensor* out_tensor {nullptr};
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out_tensor));

      if (out_tensor->NumElements() == 0) {
        // If a has shape [0, x] or b has shape [x, 0], the output shape
        // is a 0-element matrix, so there is nothing to do.
        return;
      }

      auto lut = t.matrix<float>();
      auto a_values = a.matrix<T>();
      auto b_values = b.matrix<T>(); 
      auto output_values = out_tensor->matrix<float>();

      for (size_t i {0}; i < a.dim_size(a_dim_remaining); ++i)
      {
        for (size_t j {0}; j < b.dim_size(b_dim_remaining); ++j)
        {
          output_values(i, j) = 0;
          for (size_t k {0}; k < a.dim_size(dim_pair[0].first); ++k)
          {
            auto a_idcs = transpose_a_ ? a_values(k, i) : a_values(i, k);
            auto b_idcs = transpose_b_ ? b_values(j, k) : b_values(k, j);
            output_values(i, j) += lut(a_idcs, b_idcs);
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
REGISTER_OP("MatMulLut")
  .Input("lut: float")
  .Input("activation_indices: T")
  .Input("weight_indices: T")
  .Output("product: float")
  .Attr("transpose_a: bool = false")
  .Attr("transpose_b: bool = false")
  .Attr("T: {uint8, uint16, uint32}")
  .SetShapeFn(tensorflow::shape_inference::MatMulShape);


//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("MatMulLut")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<tensorflow::uint8>("T"),
    custom_ops::MatMulLutOp<tensorflow::uint8>);

//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("MatMulLut")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<tensorflow::uint16>("T"),
    custom_ops::MatMulLutOp<tensorflow::uint16>);

//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("MatMulLut")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<tensorflow::uint32>("T"),
    custom_ops::MatMulLutOp<tensorflow::uint32>);
