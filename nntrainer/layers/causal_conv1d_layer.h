// causal_conv1d_layer.h
#ifndef __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__
#define __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__

#include <array>
#include <limits>
#include <string>
#include <vector>

#include <layer_devel.h>
#include <layer_impl.h>
#include <tensor_dim.h>


namespace nntrainer{

class CausalConv1DLayer : public LayerImpl {
public:
  CausalConv1DLayer();
  ~CausalConv1DLayer() override = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  bool supportBackwarding() const override { return false; }

   /**
   *  @brief  Move constructor of Depthwise Conv 1D Layer.
   *  @param[in] DepthwiseConv1DLayer &&
   */
  CausalConv1DLayer(CausalConv1DLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs DepthwiseConv1DLayer to be moved.
   */
  CausalConv1DLayer &operator=(CausalConv1DLayer &&rhs) = default;


  const std::string getType() const override {
    return CausalConv1DLayer::type;
  };



  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "causal_conv1d";

private:
  enum CausalConv1DParams { weight = 0 };
  static constexpr size_t SINGLE_INOUT_IDX = 0;
  static constexpr unsigned int KERNEL_SIZE = 3;
  std::tuple<props::FilterSize, props::KernelSize, props::Stride,
             props::Padding1D, props::Dilation>
    conv_props;

  std::array<unsigned int, 1> weight_idx;

  void validateInputShape(const nntrainer::TensorDim &input_dim) const;
};

}

#endif // __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__