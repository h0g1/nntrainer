// SPDX-License-Identifier: Apache-2.0
/**
* Copyright (C) 2026 Hyeong-Gwon Hong
*
* @file causal_conv1d_layer.cpp
* @date 01 April 2026
* @brief Causal depthwise Conv1D layer for CausalLM
* @see https://github.com/nntrainer/nntrainer
* @author Hyeong-Gwon Hong
* @bug No known bugs except for NYI items
*/

#include "causal_conv1d_layer.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

CausalConv1DLayer::CausalConv1DLayer() : LayerImpl() {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void CausalConv1DLayer::validateInputShape(
  const nntrainer::TensorDim &input_dim) const {

  NNTR_THROW_IF(input_dim.rank() != 4, std::invalid_argument)
    << "[CausalConv1DLayer] input rank must be 4, but got "
    << input_dim.rank();

  // DepthwiseConv1D-style layout: B x C x 1 x W
  NNTR_THROW_IF(input_dim.channel() < 1, std::invalid_argument)
    << "[CausalConv1DLayer] input channel must be positive, but got "
    << input_dim.channel();

  NNTR_THROW_IF(input_dim.height() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] input height must be 1 for BxCx1xW layout, but got "
    << input_dim.height();

  NNTR_THROW_IF(input_dim.width() < 1, std::invalid_argument)
    << "[CausalConv1DLayer] input width must be positive, but got "
    << input_dim.width();

  NNTR_THROW_IF(
    input_dim.getDataType() != ml::train::TensorDim::DataType::FP32,
    std::invalid_argument)
    << "[CausalConv1DLayer] input dtype must be FP32.";
}

void CausalConv1DLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  NNTR_THROW_IF(training, std::invalid_argument)
    << "[CausalConv1DLayer] training/backward is not supported yet.";

  NNTR_THROW_IF(to == 0 || to <= from, std::invalid_argument)
    << "[CausalConv1DLayer] invalid incremental range: from=" << from
    << ", to=" << to;

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor weight_tensor = context.getWeight(weight_idx[weight]);

  const nntrainer::TensorDim &in_dim = input.getDim();

  const unsigned int B = in_dim.batch();
  const unsigned int C = in_dim.channel();
  const unsigned int W = in_dim.width(); // seq_len

  NNTR_THROW_IF(input.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[CausalConv1DLayer] input must be FP32.";

  NNTR_THROW_IF(output.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[CausalConv1DLayer] output must be FP32.";

  NNTR_THROW_IF(to > W, std::invalid_argument)
    << "[CausalConv1DLayer] invalid incremental end: to=" << to
    << ", seq_len=" << W;

  float *input_ptr = input.getData();
  const float *weight_ptr = weight_tensor.getData();
  float *output_ptr = output.getData();

  /**
   * Tensor layout:
   *   input  : [B, C, 1, W]
   *   weight : [C, 1, 1, K]  (depthwise_conv1d-compatible public layout)
   *   output : [B, C, 1, W]
   *
   * Kernel side should interpret:
   *   B = batch
   *   C = channels
   *   to = valid sequence length to compute
   */
  nntrainer::causal_depthwise_conv1d_k3_fp32(
    input_ptr, weight_ptr, output_ptr, B, C, to);
}

void CausalConv1DLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  throw std::runtime_error(
    "[CausalConv1DLayer] forwarding() is not used. "
    "Use incremental_forwarding().");
}

void CausalConv1DLayer::finalize(nntrainer::InitLayerContext &context) {
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_regularizer = std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] requires exactly 1 input, but got "
    << context.getNumInputs();

  const nntrainer::TensorDim &input_dim = context.getInputDimensions()[0];
  validateInputShape(input_dim);

  const unsigned int channels = input_dim.channel();
  const unsigned int seq_len = input_dim.width();

  ml::train::TensorDim::TensorType weight_type(
    context.getFormat(), context.getWeightDataType());

  // DepthwiseConv1D-compatible weight layout: [C, 1, 1, K]
  nntrainer::TensorDim weight_dim(channels, 1, 1, KERNEL_SIZE, weight_type);

  nntrainer::TensorDim output_dim(
    input_dim.batch(), channels, 1, seq_len,
    {context.getFormat(), context.getActivationDataType()});
  output_dim.setDataType(ml::train::TensorDim::DataType::FP32);

  context.setOutputDimensions({output_dim});

  weight_idx[weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "causal_conv1d_weight", true);
}

void CausalConv1DLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
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

void CausalConv1DLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
}

#ifdef PLUGGABLE
extern "C" {
nntrainer::Layer *create_causal_conv1d_layer() {
  return new nntrainer::CausalConv1DLayer();
}
void destroy_causal_conv1d_layer(nntrainer::Layer *layer) { delete layer; }
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_causal_conv1d_layer,
  destroy_causal_conv1d_layer,
  nntrainer::CausalConv1DLayer::type
};
}
#endif

} // namespace nntrainer