// SPDX-License-Identifier: Apache-2.0
/**
 * @file	kai4_tensor.cpp
 * @date	12 January 2026
 * @brief	This is Kai4Tensor class for Kai library integration.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Hyeonggwon <hyeonggwon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <kai4_tensor.h>
#include <tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

#if defined(ENABLE_FP16) && defined(__aarch64__)
#include <cpu_backend/arm/kleidiai_interface.h>
#endif

#if defined(ENABLE_FP16) && defined(__aarch64__)
#include <kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h>
#include <kai_lhs_quant_pack_qsi8d32p_f32.h>
#include <kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h>
#include <cpu_backend/arm/kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h>
#endif

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nntrainer {

Kai4Tensor::Kai4Tensor(std::string name_, Tformat fm, QScheme qscheme_) : 
  TensorBase(name_, fm, Tdatatype::QINT4), qscheme(qscheme_) {
  offset = 0;
  _idx_variant = 4; //Default set to matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod
  transB = true;
}

Kai4Tensor::Kai4Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name, QScheme qscheme_) :
  TensorBase(d, false, init, name), qscheme(qscheme_) {
  // Kai4Tensor expects 2D semantics (NxK) for packing
  // Adjust based on strict requirement if needed, essentially we want 2D
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1, std::invalid_argument)
    << "Kai4Tensor must be 2 dimensional tensor (NxK) with batch size 1 and "
       "channel 1";
  
  NNTR_THROW_IF(d.width() % 32 != 0, std::invalid_argument)
      << "Kai4Tensor width must be divisible by 32 (default block size)";

  if (alloc_now) {
    allocate();
  }
  offset = 0;
  _idx_variant = 4; //Default set to matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod
  transB = true;
}

Kai4Tensor::Kai4Tensor(const TensorDim &d, const void *buf) :
  Kai4Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr) {
      // If buf is provided, we assume it is ALREADY PACKED compatible data
      // For raw packing, use pack() explicitly
      copy_kai(buf);
    }
  }
}

void Kai4Tensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    // Use calculateSize() to determine allocation size
    size_t alloc_size = size();

    mem_data = new MemoryData((void *)(new uint8_t[alloc_size]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<uint8_t>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void *Kai4Tensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint8_t>() + offset;
}

void Kai4Tensor::setZero() {
  if (!data)
    return;
  uint8_t *ptr = (uint8_t *)getData();
  std::fill(ptr, ptr + size(), 0);
}

void Kai4Tensor::initialize() {
  if (empty() || !isAllocated())
    return;
  setZero();
}

size_t Kai4Tensor::size() const {
#if defined(ENABLE_FP16) && defined(__aarch64__)
  // Compute the packed size using Kai library function
  // This requires knowing packing params: nr, kr, bl
  // If not set, we can't compute size accurately
  if (_nr == 0 || _kr == 0) {
    // Fallback: just return dimension size
    // In practice, setPackingParams or pack() should be called first
    return getDim().getDataLen();
  }
  
  // 
  return nntr_kai_get_rhs_packed_size_qsi8d32p_qsi4c32p(height(), width(), _idx_variant, transB)
#else
  return getDim().getDataLen();
#endif
}

size_t Kai4Tensor::getMemoryBytes() const {
  return size();
}

QScheme Kai4Tensor::q_scheme() const {
  return qscheme;  // Return the actual qscheme member variable
}



void Kai4Tensor::setKernelVariant(uint32_t variant_idx) {
  NNTR_THROW_IF(variant_idx > 7, std::invalid_argument)
    << "Kai4Tensor::setKernelVariant: variant_idx must be between 0 and 7, got " << variant_idx;
  
  _idx_variant = variant_idx;
  
  // Note: Packing parameters (nr, kr) will be retrieved from the kai interface
  // when needed via the ukernel_variants array, not stored here explicitly
}

void Kai4Tensor::copy_kai(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  std::memcpy(getData(), buf, size());
}

void Kai4Tensor::pack(const uint8_t *rhs, const float *bias, size_t nr,
                       size_t kr, size_t sr, size_t bl) {
#if defined(ENABLE_FP16) && defined(__aarch64__)
  _nr = nr;
  _kr = kr;
  _bl = bl;

  // Ensure allocated with correct size based on params
  if (!isAllocated()) {
      allocate(); 
  } else {
    // Note: If dimensions change significantly, manual reallocation might be needed
    // but here we assume user manages tensor lifecycle or tensor is fresh
  }

  size_t n = height();
  size_t k = width();
  size_t num_groups = 1; // Assuming 1 group for standard gemm

  // Extra bytes appends padding if needed, usually 0 for this kernel
  size_t extra_bytes = 0;

  // Params struct configuration
  struct kai_rhs_pack_qs4cxs1s0_param params;
  params.lhs_zero_point = 1; // As seen in kernel for symmetric-ish LHS
  params.rhs_zero_point = 8; // As seen in kernel for XOR offset
  
  // Call Kai packing function
  kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
    num_groups, n, k, nr, kr, sr, bl, rhs, bias, getData(), extra_bytes,
    &params);
#else
  throw std::runtime_error("Kai4Tensor::pack() requires ARM64 with FP16 support");
#endif
}

Tensor &Kai4Tensor::dot(Tensor const &input, Tensor &output, bool trans,
                         bool trans_in, float beta) const {
#if defined(ENABLE_FP16) && defined(__aarch64__)
  // Check assumptions
  // 1. This tensor (RHS/Weights) must be packed for the compatible kernel
  NNTR_THROW_IF(_nr != 4 || _kr != 8, std::invalid_argument)
      << "Kai4Tensor::dot: Incompatible packing parameters for KAI matmul. "
         "Expected nr=4, kr=8.";
         
  // 2. Input (LHS/Activations) dimensions
  // Input: [Batch, Channels] usually.
  // This Tensor: [N, K] where N=OutFeatures, K=InFeatures.
  // Standard matmul: Output [Batch, N] = Input [Batch, K] x Transpose(This [N, K])
  // So 'k' in kernel corresponds to 'K' (width of this, width of input potentially)
  
  // input batch = M
  // input width = K
  size_t m = input.batch() * input.height() * input.width() * input.channel() / width();
  // Simplified: we assume 'input' is flattened to [M, K] relative to Weight [N, K]
  // Ideally checks match: input last dimension == K
  size_t k = width();
  
  // If tensor dim is (Batch, 1, 1, K), it's clean.
  // Check compatibility
  NNTR_THROW_IF(input.size() % k != 0, std::invalid_argument)
     << "Kai4Tensor::dot: Input size not divisible by K (weight input dim)";
  
  size_t n = height();
  
  // Output dim check
  NNTR_THROW_IF(output.size() != m * n, std::invalid_argument)
     << "Kai4Tensor::dot: Output dimensions mismatch";

  // Beta check
  NNTR_THROW_IF(beta != 0.0f, std::invalid_argument)
     << "Kai4Tensor::dot: beta != 0 not supported by Kai kernel yet";

  // Packing parameters for LHS
  size_t mr = 4; // required by kernel
  size_t kr = 8; // required by kernel
  size_t sr = 2; // split ratio, typically 2 or 4. Kernel uses 2 in macros/logic usually?
                 // Checking kai_matmul_..._dotprod.c: static const size_t kai_sr = 2;
  size_t bl = 32;

  // 3. Pack LHS (Input)
  // Allocate temp buffer
  size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(
      m, k, bl, mr, kr, sr);
      
  std::vector<uint8_t> lhs_packed_buf(lhs_packed_size);
  
  const float *input_data = input.getData<float>();
  // Warning: assuming input is float. Kai4Tensor doesn't support mixed otherwise yet.
  
  // Pack LHS
  size_t m_idx_start = 0;
  // stride: number of bytes per row. Assuming contiguous float input
  size_t lhs_stride = k * sizeof(float); 
  
  kai_run_lhs_quant_pack_qsi8d32p_f32(
      m, k, bl, mr, kr, sr, m_idx_start, input_data, lhs_stride, lhs_packed_buf.data());

  // 4. Run Matmul using selected variant
  float *dst = output.getData<float>();
  float scalar_min = -std::numeric_limits<float>::infinity();
  float scalar_max = std::numeric_limits<float>::infinity();
  
  // Use the high-level interface that dispatches to the right kernel variant
  nntr_kai_gemm_qsi8d32p_qsi4c32p_olp(
      m, n, k,
      lhs_packed_buf.data(),  // LHS activation (packed)
      getData(),               // RHS weight (packed, this tensor)
      dst,                     // Output
      _idx_variant,            // Use the variant set via setKernelVariant()!
      true,                    // transB
      scalar_min,
      scalar_max);

  return output;
#else
  throw std::runtime_error("Kai4Tensor::dot() requires ARM64 with FP16 support");
#endif
}

} // namespace nntrainer
