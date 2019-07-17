
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
using tensorflow::OpKernel;
using tensorflow::OpKernelContext;
using tensorflow::OpKernelConstruction;
using tensorflow::uint8;
using tensorflow::uint16;
using tensorflow::uint32;
using tensorflow::int64;

static constexpr uint8 numRows {4};
static constexpr uint8 numCols {4};

//-----------------------------------------------------------------------------
template <typename T>
class DotProductLutOp : public OpKernel {
  public:
    using LookUpTable = std::array<float, numRows*numCols>;

    explicit DotProductLutOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    
    void Compute(OpKernelContext* ctx) override {
      // get input tensors
      const Tensor& lookupTableTensor = ctx->input(0);
      const Tensor& activationIndicesTensor = ctx->input(1);
      const Tensor& weightIndicesTensor = ctx->input(2);
      auto lookupTable = lookupTableTensor.matrix<float>();
      auto aIdcs = activationIndicesTensor.flat<T> ();
      auto wIdcs = weightIndicesTensor.flat<T> ();
      int64 nElts = activationIndicesTensor.NumElements();
      
      Tensor* output_tensor {nullptr};
      OP_REQUIRES_OK (ctx, ctx->allocate_output(0, TensorShape({1}), &output_tensor));
      auto output = output_tensor->flat<float>();
      
      output(0) = 0.f;
      for (T i {0}; i < T(nElts); ++i)
        output(0) += lookupTable(aIdcs(i), wIdcs(i));
      
    }

   static auto getLut () -> DotProductLutOp::LookUpTable {return lut_;};
   static void setLut (const LookUpTable& newLut) {lut_ = newLut;};

  private:
    float lookUp (const T rowIndex, const T colIndex) {
      return lut_[colIndex + rowIndex * numCols];
    }

    static const LookUpTable lut_;
};
//-----------------------------------------------------------------------------

template <typename T>
const std::array<float, 16> DotProductLutOp<T>::lut_ = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f};

} // custom_ops


//-----------------------------------------------------------------------------
REGISTER_OP("DotProductLut")
  .Input("lut: float")
  .Input("activation_indices: T")
  .Input("weight_indices: T")
  .Output("product: float")
  .Attr("T: {uint8, uint16, uint32}")
  .SetShapeFn(tensorflow::shape_inference::ScalarShape);


//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("DotProductLut")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<tensorflow::uint8>("T"),
    custom_ops::DotProductLutOp<tensorflow::uint8>);

//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("DotProductLut")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<tensorflow::uint16>("T"),
    custom_ops::DotProductLutOp<tensorflow::uint16>);

//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("DotProductLut")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<tensorflow::uint32>("T"),
    custom_ops::DotProductLutOp<tensorflow::uint32>);
