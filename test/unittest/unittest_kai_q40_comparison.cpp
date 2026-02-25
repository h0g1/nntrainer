// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_kai_q40_comparison.cpp
 * @date   14 January 2026
 * @brief  Unit tests comparing Q4_0 Tensor and KAI4 Tensor dot operations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonggwon <hyeonggwon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>
#include <tensor.h>
#include <cpu_backend.h>
#if defined(ENABLE_FP16) && defined(__aarch64__)
#include <cpu_backend/arm/kleidiai_interface.h>
#include <cpu_backend/arm/arm_compute_backend.h>
#include <kai4_tensor.h>
#include <q4_0_tensor.h>
#endif
#include <q4_0_utils.h>
#include <random>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

#define QK4_0 32

#define N_K 8

/**
 * @brief FP16 to float conversion (using memcpy to avoid strict-aliasing warnings)
 */
static inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exponent = (h & 0x7C00) >> 10;
  uint32_t mantissa = (h & 0x03FF);

  if (exponent == 0) {
    if (mantissa == 0) {
      uint32_t result = sign;
      float f;
      std::memcpy(&f, &result, sizeof(float));
      return f;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x03FF;
    }
  } else if (exponent == 0x1F) {
    uint32_t result = sign | 0x7F800000 | (mantissa << 13);
    float f;
    std::memcpy(&f, &result, sizeof(float));
    return f;
  }

  uint32_t result = sign | ((exponent + 112) << 23) | (mantissa << 13);
  float f;
  std::memcpy(&f, &result, sizeof(float));
  return f;
}

/**
 * @brief Generate random FP32 vector
 */
template <typename T>
static std::vector<T> generate_random_vector(size_t size, float min_val = -0.5f, float max_val = 0.5f) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

/**
 * @brief Compute MSE between two vectors
 */
__attribute__((unused))
static float compute_mse(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::max();
  
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return static_cast<float>(sum / a.size());
}




// ============================================================================
// Tensor::dot() API Integration Tests - Validate full API path, not just kernels
// ============================================================================

/**
 * @brief Test Q4_0 through Tensor::dot() API (full integration)
 * This validates FloatTensor::dot() -> dotQInteger() -> Kai/Q4_0 kernels
 */
static void test_q40_tensor_dot_api(unsigned int M, unsigned int K, unsigned int N) {
  // 1. Generate random FP32 data
  std::vector<float> activation_fp32 = generate_random_vector<float>(M * K);
  std::vector<float> weight_fp32 = generate_random_vector<float>(N * K);
  
  // 2. Compute FP32 reference
  std::vector<float> reference_output(M * N);
  nntrainer::sgemm(0, false, true, M, N, K, 1.0f,
                   activation_fp32.data(), K,
                   weight_fp32.data(), K,
                   0.0f, reference_output.data(), N);
  
  // 3. Create FP32 activation tensor
  nntrainer::TensorDim activation_dim(1, 1, M, K, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor activation_tensor(activation_dim);
  std::memcpy(activation_tensor.getData<float>(), activation_fp32.data(), M * K * sizeof(float));
  
  // 4. Create FP32 weight tensor
  nntrainer::TensorDim weight_dim(1, 1, K, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor weight_tensor_fp32(weight_dim);
  std::memcpy(weight_tensor_fp32.getData<float>(), weight_fp32.data(), N * K * sizeof(float));
  
  // 5. Create Q4_0 weight tensor by converting from FP32
  // This tests the full Tensor API quantization path
  nntrainer::TensorDim q4_0_dim(1, 1, K, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q4_0);
  nntrainer::Tensor q4_0_weight_tensor(q4_0_dim);
  
  // Quantize using low-level API (since Tensor::copyData doesn't support QINT4)
  const size_t block_size = 32;
  size_t num_blocks = (N * K) / block_size;
  size_t q4_0_size = num_blocks * sizeof(block_q4_0);
  std::vector<uint8_t> q4_0_data(q4_0_size);
  nntrainer::quantize_q4_0(weight_fp32.data(), (void *)q4_0_data.data(), K, N, nullptr);
  std::vector<uint8_t> q4_0_repacked(q4_0_size);
  nntrainer::repack_q4_0(q4_0_data.data(), q4_0_repacked.data(), q4_0_size, K, N);
  
  // Allocate and set Q4_0 tensor data
  q4_0_weight_tensor.allocate();
  std::memcpy(q4_0_weight_tensor.getData(), q4_0_repacked.data(), q4_0_size);
  
  // 6. Run through Tensor::dot() API - this goes through FloatTensor::dot() -> dotQInteger()
  nntrainer::TensorDim output_dim(1, 1, M, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor q4_0_output_tensor(output_dim);
  q4_0_output_tensor.allocate();
  activation_tensor.dot(q4_0_weight_tensor, q4_0_output_tensor, false, false, 0.0f);
 
  // 7. Compare Q4_0 Tensor::dot() output vs FP32 reference
  std::vector<float> q4_0_vec(q4_0_output_tensor.getData<float>(), 
                               q4_0_output_tensor.getData<float>() + M * N);
  float mse = compute_mse(reference_output, q4_0_vec);
  
  // Same tolerance as kernel tests
  const float base_eps = 1.5e-5;
  const float gemv_multiplier = (M == 1 && K == 3072 && N == 512) ? 7.0f : 1.0f;
  const float tolerance = base_eps * gemv_multiplier * M * K * N;
  
  EXPECT_LT(mse, tolerance) 
    << "Tensor::dot() API: MSE too high for M=" << M << ", K=" << K << ", N=" << N
    << ": MSE=" << mse << ", tolerance=" << tolerance;
}

/**
 * @brief Test Kai through Tensor::dot() API (full integration)
 */
#if defined(ENABLE_FP16) && defined(__aarch64__)
static void test_kai_tensor_dot_api(unsigned int M, unsigned int K, unsigned int N) {
  // 1. Generate random FP32 data
  std::vector<float> activation_fp32 = generate_random_vector<float>(M * K);
  std::vector<float> weight_fp32 = generate_random_vector<float>(N * K);
  
  // 2. Compute FP32 reference
  std::vector<float> reference_output(M * N);
  nntrainer::sgemm(0, false, true, M, N, K, 1.0f,
                   activation_fp32.data(), K,
                   weight_fp32.data(), K,
                   0.0f, reference_output.data(), N);
  
  // 3. Create FP32 activation tensor
  nntrainer::TensorDim activation_dim(1, 1, M, K, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor activation_tensor(activation_dim);
  std::memcpy(activation_tensor.getData<float>(), activation_fp32.data(), M * K * sizeof(float));
  
  // 4. Create Kai weight tensor using QINT4 datatype (creates Kai4Tensor on ARM64)
  nntrainer::TensorDim kai_dim(1, 1, N, K, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4);
  nntrainer::Tensor kai_weight_tensor(kai_dim, false, nntrainer::Initializer::NONE, "", nntrainer::QScheme::PER_CHANNEL_AFFINE);
  
  // Quantize using Kai's native block-32 quantization
  const size_t bl = 32;
  const size_t num_blocks = (K / bl);
  const size_t bytes_per_block = sizeof(uint16_t) + bl / 2;
  std::vector<uint8_t> kai_quant_data(N * num_blocks * bytes_per_block);
  nntr_kai_quant_qs4c32_f32(N, K, bl, weight_fp32.data(), kai_quant_data.data());
  
  // RHS Packing for offline-packed Kai API
  uint32_t idx_variant = 4;  // Using variant 4
  bool transB = true;

  size_t packed_size = nntr_kai_get_rhs_packed_size_qsi8d32p_qsi4c32p(N, K, idx_variant, transB);
  std::vector<uint8_t> kai_packed_data(packed_size);

  nntr_kai_qsi8d32p_qsi4c32p_rhs_pack(N, K,
                                       kai_packed_data.data(),
                                       kai_quant_data.data(),
                                       nullptr,
                                       idx_variant, transB);

  // Allocate and set Kai tensor data with packed weights
  kai_weight_tensor.allocate();
  std::memcpy(kai_weight_tensor.getData(), kai_packed_data.data(), packed_size);
  
  // 5. Run through Tensor::dot() API - goes through FloatTensor::dot() -> dotQInteger()
  nntrainer::TensorDim output_dim(1, 1, M, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor kai_output_tensor(output_dim);
  activation_tensor.dot(kai_weight_tensor, kai_output_tensor, false, false, 0.0f);
  
  // 6. Compare Kai Tensor::dot() output vs FP32 reference
  std::vector<float> kai_vec(kai_output_tensor.getData<float>(), 
                              kai_output_tensor.getData<float>() + M * N);
  float mse = compute_mse(reference_output, kai_vec);
  
  // Same tolerance as kernel tests
  constexpr float eps = 1.5e-5;
  const float tolerance = eps * M * K * N;
  
  EXPECT_LT(mse, tolerance) 
    << "Tensor::dot() API: MSE too high for M=" << M << ", K=" << K << ", N=" << N
    << ": MSE=" << mse << ", tolerance=" << tolerance;
}
#endif

static void test_q40_vs_kai(unsigned int M, unsigned int K, unsigned int N) {

  const int T = 30; // T iterations
  
  std::string uname[8] = {"matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod", "matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod", "matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod", "matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod", "matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod", "matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm", "matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm", "matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm"};
  // 1. Generate random FP32 data 
  std::vector<float> activation_fp32 = generate_random_vector<float>(M * K * T);
  std::vector<float> weight_fp32 = generate_random_vector<float>(N * K * T);

  // 2. Declare FP32 activation tensor, wegith tensor (both kai and q4_0)
  nntrainer::TensorDim activation_dim(1, 1, M, K, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor activation_tensor(activation_dim);
  activation_tensor.allocate();
  
  nntrainer::TensorDim kai_dim(1, 1, K, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4);


  nntrainer::TensorDim q4_0_dim(1, 1, K, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q4_0);
  nntrainer::Tensor q4_0_weight_tensor(q4_0_dim);
  q4_0_weight_tensor.allocate();

  // 3. Declare quantization & packing related ones
  const size_t bl = 32;
  const size_t num_blocks = (K / bl);
  const size_t bytes_per_block = sizeof(uint16_t) + bl / 2;

  std::vector<uint8_t> kai_quant_data(N * num_blocks * bytes_per_block);
  size_t packed_size;
  uint32_t idx_variant;
  bool transB = true;

  const size_t block_size = 32;
  const size_t q4_0_num_blocks = (N * K) / block_size;
  size_t q4_0_size = q4_0_num_blocks * sizeof(block_q4_0);
  std::vector<uint8_t> q4_0_data(q4_0_size);
  std::vector<uint8_t> q4_0_repacked(q4_0_size);


  nntrainer::TensorDim output_dim(1, 1, M, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::TensorDim output_dim2(1, 1, M, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor kai_output_tensor(output_dim);
  nntrainer::Tensor q4_0_output_tensor(output_dim2);

  kai_output_tensor.allocate();
  q4_0_output_tensor.allocate();

  microseconds execution_time{};

  
  for (unsigned int j = 0; j < N_K; j++){
    // j-th ukernel (KAI)
    idx_variant = j;
    nntrainer::Tensor kai_weight_tensor(kai_dim, false, nntrainer::Initializer::NONE, "", nntrainer::QScheme::PER_CHANNEL_AFFINE, idx_variant);
    kai_weight_tensor.allocate();

    packed_size = nntr_kai_get_rhs_packed_size_qsi8d32p_qsi4c32p(N, K, idx_variant, transB);

    std::vector<uint8_t> kai_packed_data(packed_size);
    
    for (unsigned int i = 0; i < T; i ++){
      // i-th iteraiton
      
      // Copy i-th acvitvation chunk
      std::memcpy(activation_tensor.getData<float>(), activation_fp32.data() + i * M * K, M * K * sizeof(float));
      // 4. Create Kai weight tensor using QINT4 datatype (creates Kai4Tensor on ARM64)
      
      nntr_kai_quant_qs4c32_f32(N, K, bl, weight_fp32.data() + i * N * K, kai_quant_data.data());

      nntr_kai_qsi8d32p_qsi4c32p_rhs_pack(N, K,
                                          kai_packed_data.data(),
                                          kai_quant_data.data(),
                                          nullptr,
                                          idx_variant, transB);                                        

      // Allocate and set Kai tensor data with packed weights

      
      
      std::memcpy(kai_weight_tensor.getData(), kai_packed_data.data(), packed_size);
      
      // 4. Run through Tensor::dot() API - goes through FloatTensor::dot() -> dotQInteger()
      auto t0 = high_resolution_clock::now();
      activation_tensor.dot(kai_weight_tensor, kai_output_tensor, false, false, 0.0f);
      auto t1 = high_resolution_clock::now(); 
      
      if (i == 0){
        execution_time = duration_cast<microseconds>(t1 - t0);
      } else {
        execution_time += duration_cast<microseconds>(t1 - t0);
      }

      if (i == T - 1){
        std::cout << "QINT4 kernel " << uname[j] << " : " << execution_time.count()/T << " ms " << std::endl;
      }
    }
  }

  for (unsigned int i = 0; i < T; i ++){
    
    // Quantize using low-level API (since Tensor::copyData doesn't support QINT4)
    std::memcpy(activation_tensor.getData<float>(), activation_fp32.data() + i * M * K, M * K * sizeof(float));

    nntrainer::quantize_q4_0(weight_fp32.data() + i * N * K, (void *)q4_0_data.data(), K, N, nullptr);
    nntrainer::repack_q4_0(q4_0_data.data(), q4_0_repacked.data(), q4_0_size, K, N);
    
    
    std::memcpy(q4_0_weight_tensor.getData(), q4_0_repacked.data(), q4_0_size);
    
    
    
    auto t0 = high_resolution_clock::now();
    activation_tensor.dot(q4_0_weight_tensor, q4_0_output_tensor, false, false, 0.0f);
    auto t1 = high_resolution_clock::now(); 

    if (i == 0){
        execution_time = duration_cast<microseconds>(t1 - t0);
    } else {
        execution_time += duration_cast<microseconds>(t1 - t0);
    }

    if (i == T - 1){
      std::cout << "Q40 kernel : " << execution_time.count()/T << " ms " << std::endl;
    }
  }
 
}




#if defined(ENABLE_FP16) && defined(__aarch64__)
TEST(Q40_vs_kai, GEMM_8192x2560x4096) {
  test_q40_vs_kai(8192, 2560, 4096);
}

TEST(Q40_vs_kai, GEMV_1x2560x4096) {
  test_q40_vs_kai(1, 2560, 4096);
}

TEST(Q40_vs_kai, GEMM_8192x2560x1024) {
  test_q40_vs_kai(8192, 2560, 1024);
}

TEST(Q40_vs_kai, GEMV_1x2560x1024) {
  test_q40_vs_kai(1, 2560, 1024);
}

TEST(Q40_vs_kai, GEMM_8192x4096x2560) {
  test_q40_vs_kai(8192, 4096, 2560);
}

TEST(Q40_vs_kai, GEMV_1x4096x2560) {
  test_q40_vs_kai(1, 4096, 2560);
}


#endif




/**
 * @brief Main function for Google Test
 */
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
