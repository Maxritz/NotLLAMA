#include "vulkan/vk_backend.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

static float rand_float() { return (float)(rand() % 1000) / 100.0f - 5.0f; }

// CPU fp16→f32 matching GPU shader
static float fp16_to_f32(uint32_t bits) {
    uint32_t s = (bits >> 15) & 1, e = (bits >> 10) & 0x1F, m = bits & 0x3FF;
    if (e == 0) return 0.0f;
    uint32_t f32 = (s << 31) | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f32, 4); return r;
}

// CPU Q4_0 dequant for one column
static float dequant_q4_0(const uint8_t* qdata, uint32_t k) {
    uint32_t block = k / 32, sub = k % 32;
    uint32_t q_off = block * 18;
    uint32_t d_bits = qdata[q_off] | (qdata[q_off + 1] << 8);
    float d = fp16_to_f32(d_bits);
    uint8_t byte_val = qdata[q_off + 2 + sub / 2];
    uint8_t nibble = (sub % 2 == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
    return d * (float)((int)nibble - 8);
}

static void quantize_q4_0(const float* src, uint8_t* dst, uint32_t n) {
    uint32_t num_blocks = (n + 31) / 32;
    for (uint32_t b = 0; b < num_blocks; b++) {
        uint32_t base = b * 32;
        uint32_t remain = std::min(32u, n - base);
        float max_abs = 0;
        for (uint32_t i = 0; i < remain; i++) max_abs = std::max(max_abs, std::abs(src[base + i]));
        float d = (max_abs > 0) ? max_abs / 7.0f : 1e-10f;
        uint32_t f32bits; memcpy(&f32bits, &d, 4);
        uint32_t sign = (f32bits >> 31) & 1;
        int exp = (int)((f32bits >> 23) & 0xFF) - 127;
        uint32_t mant = f32bits & 0x7FFFFF;
        if (exp > 15) { exp = 15; mant = 0; } if (exp < -14) { exp = -14; mant = 0; }
        uint16_t fp16 = (uint16_t)((sign << 15) | ((uint16_t)(exp + 15) << 10) | (uint16_t)(mant >> 13));
        dst[b * 18] = fp16 & 0xFF; dst[b * 18 + 1] = fp16 >> 8;
        for (uint32_t i = 0; i < 32; i++) {
            float val = (i < remain) ? src[base + i] : 0;
            int q = std::max(-8, std::min(7, (int)roundf(val / d)));
            uint8_t nibble = (uint8_t)(q + 8);
            uint32_t byte_idx = b * 18 + 2 + i / 2;
            if (i % 2 == 0) dst[byte_idx] = (dst[byte_idx] & 0xF0) | nibble;
            else dst[byte_idx] = (dst[byte_idx] & 0x0F) | (nibble << 4);
        }
    }
}

int main() {
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    VkApplicationInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_4;
    ici.pApplicationInfo = &ai;
    VkInstance inst;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return 1;

    VulkanBackend bk;
    if (!bk.init(inst) || !bk.load_pipelines("shaders_spv")) return 1;

    uint32_t M = 3, N = 7, K = 64, num_blocks = (K + 31) / 32;
    std::vector<float> A(M * K), B_f32(K * N);
    for (auto& x : A) x = rand_float();
    for (auto& x : B_f32) x = rand_float();

    // Quantize B to Q4_0
    uint32_t q_size = N * num_blocks * 18;
    std::vector<uint8_t> B_q(q_size);
    for (uint32_t col = 0; col < N; col++) {
        std::vector<float> col_data(K);
        for (uint32_t k = 0; k < K; k++) col_data[k] = B_f32[k * N + col];
        quantize_q4_0(col_data.data(), B_q.data() + col * num_blocks * 18, K);
    }

    // Compute CPU Q4_0 GEMM (same algorithm as shader)
    std::vector<float> C_cpu(M * N, 0);
    for (uint32_t r = 0; r < M; r++)
        for (uint32_t c = 0; c < N; c++)
            for (uint32_t k = 0; k < K; k++)
                C_cpu[r * N + c] += A[r * K + k] * dequant_q4_0(B_q.data() + c * num_blocks * 18, k);

    // Upload & run GPU Q4_0 GEMM
    auto up = [&](const void* d, uint32_t bytes) {
        GpuBuffer buf = bk.create_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        bk.upload_to_buffer(buf, d, bytes); return buf;
    };
    GpuBuffer gA = up(A.data(), M * K * 4);
    GpuBuffer gB_q4 = up(B_q.data(), q_size);
    GpuBuffer gC = bk.create_buffer(M * N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    bk.begin_batch(); bk.gemm_q4_0(gA, gB_q4, gC, M, N, K, num_blocks); bk.end_batch();

    std::vector<float> C_gpu(M * N);
    bk.download_from_buffer(gC, C_gpu.data(), M * N * 4);

    // Compare GPU Q4_0 vs CPU Q4_0 (should be identical algorithm)
    float max_diff = 0;
    for (uint32_t i = 0; i < M * N; i++) {
        float diff = std::abs(C_gpu[i] - C_cpu[i]);
        max_diff = std::max(max_diff, diff);
    }
    bool pass = max_diff < 1e-4f;
    printf("Q4_0 GEMM: GPU vs CPU max_diff=%.6f\n", max_diff);
    printf("%s\n", pass ? "PASS" : "FAIL");

    // Also compute F32 GEMM for reference
    std::vector<float> C_f32(M * N, 0);
    for (uint32_t r = 0; r < M; r++)
        for (uint32_t c = 0; c < N; c++)
            for (uint32_t k = 0; k < K; k++)
                C_f32[r * N + c] += A[r * K + k] * B_f32[k * N + c];
    float max_re = 0;
    for (uint32_t i = 0; i < M * N; i++) {
        float re = std::abs(C_cpu[i] - C_f32[i]) / std::max(std::abs(C_f32[i]), 1e-6f);
        max_re = std::max(max_re, re);
    }
    printf("Q4_0 vs F32 MaxRE=%.4f (quantization noise, not a bug)\n", max_re);

    auto del = [&](GpuBuffer& b) { if (b.buffer) bk.destroy_buffer(b); };
    del(gA); del(gB_q4); del(gC); bk.cleanup(); vkDestroyInstance(inst, nullptr);
    return pass ? 0 : 1;
}
