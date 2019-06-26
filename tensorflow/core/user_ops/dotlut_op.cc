
#include <array>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"

/* #include "tensorflow/core/user_ops/dotlut_op.h" */

using namespace tensorflow;

//-----------------------------------------------------------------------------
static constexpr uint8 numRows {4};
static constexpr uint8 numCols {4};

//-----------------------------------------------------------------------------
template <typename T>
class DotLutOp : public OpKernel {
  public:
    using LookUpTable = std::array<float, numRows*numCols>;

    explicit DotLutOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    
    void Compute(OpKernelContext* ctx) override {
      // get input tensors
      const Tensor& activationIndicesTensor = ctx->input(0);
      const Tensor& weightIndicesTensor = ctx->input(1);
      auto aIdcs = activationIndicesTensor.flat<T> ();
      auto wIdcs = weightIndicesTensor.flat<T> ();
      int64 nElts = activationIndicesTensor.NumElements();
      
      Tensor* output_tensor {nullptr};
      OP_REQUIRES_OK (ctx, ctx->allocate_output(0, TensorShape({1}), &output_tensor));
      auto output = output_tensor->flat<float>();
      
      output(0) = 0.f;
      for (T i {0}; i < T(nElts); ++i)
        output(0) += lookUp(aIdcs(i), wIdcs(i));
    }

   static auto getLut () -> DotLutOp::LookUpTable {return lut_;};
   static void setLut (const LookUpTable& newLut) {lut_ = newLut;};

  private:
    float lookUp (const T rowIndex, const T colIndex) {
      return lut_[colIndex + rowIndex * numCols];
    }

    static const LookUpTable lut_;
};
//-----------------------------------------------------------------------------

template <typename T>
const std::array<float, 16> DotLutOp<T>::lut_ = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f};

//-----------------------------------------------------------------------------
REGISTER_OP("DotLUT")
  .Input("activation_indices: T")
  .Input("weight_indices: T")
  .Output("product: float")
  .Attr("T: {uint8, uint16, uint32}")
  .SetShapeFn(shape_inference::ScalarShape);


//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("DotLUT")
    .Device(DEVICE_CPU)
    .TypeConstraint<uint8>("T"),
    DotLutOp<uint8>);

//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("DotLUT")
    .Device(DEVICE_CPU)
    .TypeConstraint<uint16>("T"),
    DotLutOp<uint16>);

//-----------------------------------------------------------------------------
REGISTER_KERNEL_BUILDER(
    Name("DotLUT")
    .Device(DEVICE_CPU)
    .TypeConstraint<uint32>("T"),
    DotLutOp<uint32>);
