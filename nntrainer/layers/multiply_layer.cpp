// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   multiply_layer.cpp
 * @date   10 Oct 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is multiply layer class (operation layer)
 *
 */

#include <multiply_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>
#include <chrono>
#include <iostream>

namespace nntrainer {

void MultiplyLayer::finalize(InitLayerContext &context) {
  auto const &input_dims = context.getInputDimensions();
  TensorDim out_dim = input_dims[0];

  if (input_dims.size() > 1) {
    TensorDim dim1 = input_dims[1];
    // Compute broadcast output shape: for each dimension, take the max
    // when one of the inputs has size 1 (standard broadcasting rules).
    for (unsigned int i = 0; i < 4; ++i) {
      if (out_dim[i] != dim1[i]) {
        if (out_dim[i] == 1) {
          out_dim.setTensorDim(i, dim1[i]);
        } else if (dim1[i] != 1) {
          throw std::invalid_argument(
            "MultiplyLayer: incompatible shapes for broadcasting at dim " +
            std::to_string(i) + " (" + std::to_string(out_dim[i]) + " vs " +
            std::to_string(dim1[i]) + ")");
        }
      }
    }
  }

  context.setOutputDimensions({out_dim});
}

void MultiplyLayer::forwarding_operation(const Tensor &input0,
                                         const Tensor &input1, Tensor &hidden) {
  input0.multiply(input1, hidden);
}

void MultiplyLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from,
                                           unsigned int to,
                                           bool training) {
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  NNTR_THROW_IF(to <= from, std::invalid_argument)
    << "MultiplyLayer::incremental_forwarding requires to > from";

  Tensor input0 = context.getInput(0);
  Tensor input1 = context.getInput(1);
  Tensor output = context.getOutput(0);

  const TensorDim &d0 = input0.getDim();
  const TensorDim &d1 = input1.getDim();
  const TensorDim &do_ = output.getDim();

  // Generic safe fallback:
  // height slicing with a single shared contiguous tensor is only straightforward
  // when batch == 1 and channel == 1.
  if (d0.batch() != 1 || d0.channel() != 1 || do_.batch() != 1 ||
      do_.channel() != 1) {
    forwarding(context, training);
    return;
  }

  auto slice_height_nchw_bc11 =
    [from, to](Tensor &t) -> Tensor {
      const TensorDim &d = t.getDim();

      // Broadcast on height: keep as-is
      if (d.height() == 1) {
        return t;
      }

      NNTR_THROW_IF(to > d.height(), std::invalid_argument)
        << "MultiplyLayer::incremental_forwarding range exceeds tensor height";

      // For [1,1,H,W], height offset is simply from * W elements.
      size_t offset = static_cast<size_t>(from) * d.width();

      return t.getSharedDataTensor(
        TensorDim(1, 1, to - from, d.width()),
        offset);
    };

  Tensor sliced_input0 = slice_height_nchw_bc11(input0);
  Tensor sliced_input1 = slice_height_nchw_bc11(input1);

  NNTR_THROW_IF(do_.height() != 1 && to > do_.height(), std::invalid_argument)
    << "MultiplyLayer::incremental_forwarding range exceeds output height";

  Tensor sliced_output;
  if (do_.height() == 1) {
    sliced_output = output;
  } else {
    size_t out_offset = static_cast<size_t>(from) * do_.width();
    sliced_output = output.getSharedDataTensor(
      TensorDim(1, 1, to - from, do_.width()),
      out_offset);
  }

  forwarding_operation(sliced_input0, sliced_input1, sliced_output);
  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

  std::chrono::microseconds micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "multiply" << micro.count() << std::endl;
}

void MultiplyLayer::calcDerivative(RunLayerContext &context) {
  context.getOutgoingDerivative(0).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(context.getInput(1)));

  context.getOutgoingDerivative(1).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(context.getInput(0)));
}

void MultiplyLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, multiply_props);
  if (!remain_props.empty()) {
    std::string msg = "[MultiplyLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
