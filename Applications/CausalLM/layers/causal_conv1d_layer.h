// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeong-Gwon Hong <h0g1.hong@samsung.com>
 *
 * @file   causal_conv1d_layer.h
 * @date   01 April 2026
 * @brief  Causal depthwise Conv1D layer for CausalLM models.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeong-Gwon Hong <h0g1.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__
#define __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <layer_impl.h>
#include <tensor_dim.h>

/**
 * @brief CausalLM custom layer namespace.
 */
namespace causallm {

/**
 * @class   CausalConv1DLayer
 * @brief   Depthwise causal Conv1D layer with kernel size 3.
 *
 * The layer expects Bx1xHxW tensors and computes each time row from the
 * current and two previous rows. It is inference-only and keeps the previous
 * two rows as state during incremental execution.
 */
WIN_EXPORT class CausalConv1DLayer final : public nntrainer::LayerImpl {
public:
  /**
   * @brief Constructor of CausalConv1DLayer.
   */
  WIN_EXPORT CausalConv1DLayer();

  /**
   * @brief Destructor of CausalConv1DLayer.
   */
  WIN_EXPORT ~CausalConv1DLayer() override = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return CausalConv1DLayer::type;
  }

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::updateTensorsByInputDimensions(RunLayerContext &context,
   * std::vector<TensorDim> input_dimensions)
   */
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "causal_conv1d";

private:
  enum CausalConv1DParams { weight = 0 };
  enum CausalConv1DTensors { conv_state = 0 };
  static constexpr size_t SINGLE_INOUT_IDX = 0;
  static constexpr unsigned int KERNEL_SIZE = 3;

  std::array<unsigned int, 1> weight_idx;
  std::array<unsigned int, 1> tensor_idx;

  void validateInputShape(const nntrainer::TensorDim &input_dim) const;
  void runLocal(nntrainer::RunLayerContext &context, unsigned int step_size,
                bool reset_state, bool training);
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__ */
