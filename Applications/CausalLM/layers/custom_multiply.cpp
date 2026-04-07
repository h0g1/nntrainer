// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   custom_multiply.cpp
 * @date   02 April 2026
 * @brief  Custom multiply layer for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <custom_multiply.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>
#include <iostream>
#include <chrono>

namespace causallm {

void CustomMultiplyLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "CustomMultiplyLayer requires exactly 2 inputs";

  const auto &input_dims = context.getInputDimensions();
  nntrainer::TensorDim out_dim = input_dims[0];
  nntrainer::TensorDim dim1 = input_dims[1];

  // Same broadcasting rule as MultiplyLayer
  for (unsigned int i = 0; i < ml::train::TensorDim::MAXDIM; ++i) {
    if (out_dim[i] != dim1[i]) {
      if (out_dim[i] == 1) {
        out_dim.setTensorDim(i, dim1[i]);
      } else if (dim1[i] != 1) {
        throw std::invalid_argument(
          "CustomMultiplyLayer: incompatible shapes for broadcasting at dim " +
          std::to_string(i) + " (" + std::to_string(out_dim[i]) + " vs " +
          std::to_string(dim1[i]) + ")");
      }
    }
  }

  context.setOutputDimensions({out_dim});
}

void CustomMultiplyLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  nntrainer::Tensor &in0 = context.getInput(INPUT_IDX_0);
  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  in0.multiply(in1, out);
}



void CustomMultiplyLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &in0 = context.getInput(INPUT_IDX_0);
  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  const auto &out_dim = out.getDim();
  const auto &d0 = in0.getDim();
  const auto &d1 = in1.getDim();

  /**
   * Assumption:
   * - incremental range [from, to) is applied on the last dimension (T / width)
   * - CausalLM path tensor layout is [B, C, 1, T]
   *
   * This matches the reason incremental forwarding exists in nntrainer:
   * only the requested step range should be processed. 0
   */

  const unsigned int bsz = out_dim.batch();
  const unsigned int ch = out_dim.channel();
  const unsigned int h = out_dim.height();
  const unsigned int w = out_dim.width();

  NNTR_THROW_IF(from > to || to > w, std::invalid_argument)
    << "CustomMultiplyLayer::incremental_forwarding invalid range: from="
    << from << " to=" << to << " width=" << w;

  if (from == to) {
    return;
  }

  for (unsigned int b = 0; b < bsz; ++b) {
    const unsigned int b0 = (d0.batch() == 1) ? 0 : b;
    const unsigned int b1 = (d1.batch() == 1) ? 0 : b;

    for (unsigned int c = 0; c < ch; ++c) {
      const unsigned int c0 = (d0.channel() == 1) ? 0 : c;
      const unsigned int c1 = (d1.channel() == 1) ? 0 : c;

      for (unsigned int hh = 0; hh < h; ++hh) {
        const unsigned int h0 = (d0.height() == 1) ? 0 : hh;
        const unsigned int h1 = (d1.height() == 1) ? 0 : hh;

        for (unsigned int ww = from; ww < to; ++ww) {
          const unsigned int w0 = (d0.width() == 1) ? 0 : ww;
          const unsigned int w1 = (d1.width() == 1) ? 0 : ww;

          float v0 = in0.getValue<float>(b0, c0, h0, w0);
          float v1 = in1.getValue<float>(b1, c1, h1, w1);

          out.setValue(b, c, hh, ww, v0 * v1);
        }
      }
    }
  }
}

void CustomMultiplyLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  context.getOutgoingDerivative(INPUT_IDX_0).copy(
    context.getIncomingDerivative(OUT_IDX)
      .multiply(context.getInput(INPUT_IDX_1)));

  context.getOutgoingDerivative(INPUT_IDX_1).copy(
    context.getIncomingDerivative(OUT_IDX)
      .multiply(context.getInput(INPUT_IDX_0)));
}

void CustomMultiplyLayer::setProperty(
  const std::vector<std::string> &values) {
  NNTR_THROW_IF(!values.empty(), std::invalid_argument)
    << "[CustomMultiplyLayer] Unknown Layer Properties count "
    << std::to_string(values.size());
}

#ifdef PLUGGABLE

nntrainer::Layer *create_custom_multiply_layer() {
  auto layer = new CustomMultiplyLayer();
  return layer;
}

void destroy_custom_multiply_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_custom_multiply_layer,
  destroy_custom_multiply_layer};
}

#endif

} // namespace causallm
