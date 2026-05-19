// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeong-Gwon Hong <h0g1.hong@samsung.com>
 *
 * @file   causal_conv1d_layer.cpp
 * @date   01 April 2026
 * @brief  Causal depthwise Conv1D layer for CausalLM models.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeong-Gwon Hong <h0g1.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "causal_conv1d_layer.h"

#include <stdexcept>

#include <compute_ops.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>

namespace causallm {

namespace {

bool isRawFp16WeightType(ml::train::TensorDim::DataType dtype) {
  return dtype == ml::train::TensorDim::DataType::FP16 ||
         dtype == ml::train::TensorDim::DataType::UINT16;
}

void causalDepthwiseConv1dK3Fp32(const float *input, const float *weight,
                                 float *output, unsigned int batch,
                                 unsigned int height, unsigned int width,
                                 unsigned int from, unsigned int to) {
  const float *w0 = weight;
  const float *w1 = weight + width;
  const float *w2 = weight + 2 * width;

  for (unsigned int b = 0; b < batch; ++b) {
    const float *x_base = input + static_cast<size_t>(b) * height * width;
    float *y_base = output + static_cast<size_t>(b) * height * width;

    for (unsigned int c = 0; c < width; ++c) {
      float prev1 =
        from > 0 ? x_base[static_cast<size_t>(from - 1) * width + c] : 0.0f;
      float prev2 =
        from > 1 ? x_base[static_cast<size_t>(from - 2) * width + c] : 0.0f;

      for (unsigned int t = from; t < to; ++t) {
        const size_t idx = static_cast<size_t>(t) * width + c;
        const float cur = x_base[idx];
        y_base[idx] = cur * w0[c] + prev1 * w1[c] + prev2 * w2[c];
        prev2 = prev1;
        prev1 = cur;
      }
    }
  }
}

} // namespace

CausalConv1DLayer::CausalConv1DLayer() : LayerImpl() {
  weight_idx.fill(std::numeric_limits<unsigned int>::max());
}

void CausalConv1DLayer::validateInputShape(
  const nntrainer::TensorDim &input_dim) const {
  NNTR_THROW_IF(input_dim.rank() != 4, std::invalid_argument)
    << "[CausalConv1DLayer] input rank must be 4, but got " << input_dim.rank();

  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] input channel must be 1 for Bx1xHxW layout, but "
       "got "
    << input_dim.channel();

  NNTR_THROW_IF(input_dim.height() < 1 || input_dim.width() < 1,
                std::invalid_argument)
    << "[CausalConv1DLayer] invalid input shape: H and W must be positive.";

  NNTR_THROW_IF(input_dim.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[CausalConv1DLayer] input dtype must be FP32.";
}

void CausalConv1DLayer::finalize(nntrainer::InitLayerContext &context) {
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] requires exactly 1 input, but got "
    << context.getNumInputs();

  const nntrainer::TensorDim &input_dim = context.getInputDimensions()[0];
  validateInputShape(input_dim);

  const auto weight_dtype = context.getWeightDataType();
  NNTR_THROW_IF(weight_dtype != ml::train::TensorDim::DataType::FP32 &&
                  !isRawFp16WeightType(weight_dtype),
                std::invalid_argument)
    << "[CausalConv1DLayer] weight dtype must be FP32, FP16, or UINT16.";

  nntrainer::TensorDim output_dim = input_dim;
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  output_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  context.setOutputDimensions({output_dim});

  ml::train::TensorDim::TensorType weight_type(context.getFormat(),
                                               context.getWeightDataType());
  nntrainer::TensorDim weight_dim(1, 1, KERNEL_SIZE, input_dim.width(),
                                  weight_type);

  weight_idx[weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "causal_conv1d_weight", true);
}

void CausalConv1DLayer::runRange(nntrainer::RunLayerContext &context,
                                 unsigned int from, unsigned int to,
                                 bool training) {
  NNTR_THROW_IF(training, std::invalid_argument)
    << "[CausalConv1DLayer] training/backward is not supported yet.";

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &weight_tensor = context.getWeight(weight_idx[weight]);

  const nntrainer::TensorDim &input_dim = input.getDim();
  validateInputShape(input_dim);

  NNTR_THROW_IF(output.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[CausalConv1DLayer] output dtype must be FP32.";

  NNTR_THROW_IF(from > to || to > input_dim.height(), std::invalid_argument)
    << "[CausalConv1DLayer] invalid range: from=" << from << ", to=" << to
    << ", H=" << input_dim.height();

  if (from == to) {
    return;
  }

  const unsigned int batch = input_dim.batch();
  const unsigned int height = input_dim.height();
  const unsigned int width = input_dim.width();

  const auto weight_dtype = weight_tensor.getDataType();
  if (weight_dtype == ml::train::TensorDim::DataType::FP32) {
    causalDepthwiseConv1dK3Fp32(
      input.getData<float>(), weight_tensor.getData<float>(),
      output.getData<float>(), batch, height, width, from, to);
    return;
  }

  NNTR_THROW_IF(!isRawFp16WeightType(weight_dtype), std::invalid_argument)
    << "[CausalConv1DLayer] weight dtype must be FP32, FP16, or UINT16.";

  nntrainer::ComputeOps *ops = context.getComputeOps();
  if (ops == nullptr) {
    ops = nntrainer::getComputeOps();
  }

  ops->causal_depthwise_conv1d_k3_fp16(
    input.getData<float>(), weight_tensor.getData<uint16_t>(),
    output.getData<float>(), batch, height, width, from, to);
}

void CausalConv1DLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  runRange(context, 0, context.getInput(SINGLE_INOUT_IDX).height(), training);
}

void CausalConv1DLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  runRange(context, from, to, training);
}

void CausalConv1DLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "[CausalConv1DLayer] calcDerivative() is not implemented. "
    "This layer is inference-only for now.");
}

void CausalConv1DLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "[CausalConv1DLayer] calcGradient() is not implemented. "
    "This layer is inference-only for now.");
}

void CausalConv1DLayer::exportTo(nntrainer::Exporter &exporter,
                                 const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
}

void CausalConv1DLayer::setProperty(const std::vector<std::string> &values) {
  NNTR_THROW_IF(!values.empty(), std::invalid_argument)
    << "[CausalConv1DLayer] does not take properties.";
}

} // namespace causallm

#ifdef PLUGGABLE
extern "C" {

nntrainer::Layer *create_causal_conv1d_layer() {
  return new causallm::CausalConv1DLayer();
}

void destroy_causal_conv1d_layer(nntrainer::Layer *layer) { delete layer; }

nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_causal_conv1d_layer, destroy_causal_conv1d_layer,
  causallm::CausalConv1DLayer::type};
}
#endif
