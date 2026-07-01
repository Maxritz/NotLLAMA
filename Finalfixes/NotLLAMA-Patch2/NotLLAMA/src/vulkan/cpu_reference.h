#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

// CPU reference implementations matching GPU shader semantics

inline void ref_gemm(const float* A, const float* B, float* C,
                      uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            float sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
}

inline void ref_rms_norm(const float* x, const float* w, float* y,
                          uint32_t dim, float eps, uint32_t rows) {
    for (uint32_t r = 0; r < rows; r++) {
        float ss = 0;
        for (uint32_t i = 0; i < dim; i++)
            ss += x[r * dim + i] * x[r * dim + i];
        float scale = 1.0f / std::sqrt(ss / dim + eps);
        for (uint32_t i = 0; i < dim; i++)
            y[r * dim + i] = x[r * dim + i] * scale * w[i];
    }
}

inline void ref_rope(float* data, uint32_t head_dim, uint32_t num_heads,
                      uint32_t position, float theta_base) {
    for (uint32_t h = 0; h < num_heads; h++) {
        for (uint32_t d = 0; d < head_dim; d += 2) {
            float inv_freq = 1.0f / std::pow(theta_base, (float)d / (float)head_dim);
            float angle = (float)position * inv_freq;
            float c = std::cos(angle), s = std::sin(angle);
            float* row = data + h * head_dim;
            float x0 = row[d], x1 = row[d + 1];
            row[d] = x0 * c - x1 * s;
            row[d + 1] = x0 * s + x1 * c;
        }
    }
}

inline float ref_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

inline void ref_silu_mul(const float* gate, const float* up, float* out, uint32_t count) {
    for (uint32_t i = 0; i < count; i++)
        out[i] = ref_silu(gate[i]) * up[i];
}

inline void ref_attention_scores(const float* Q, const float* K_cache,
                                  float* scores, uint32_t head_dim,
                                  uint32_t seq_len, uint32_t max_seq,
                                  float scale, uint32_t num_q_heads,
                                  uint32_t gqa_ratio) {
    for (uint32_t h = 0; h < num_q_heads; h++) {
        uint32_t kv_h = h / gqa_ratio;
        for (uint32_t p = 0; p < seq_len; p++) {
            float sum = 0;
            for (uint32_t d = 0; d < head_dim; d++)
                sum += Q[h * head_dim + d] * K_cache[kv_h * max_seq * head_dim + p * head_dim + d];
            scores[h * max_seq + p] = sum * scale;
        }
    }
}

inline void ref_softmax(float* data, uint32_t seq_len, uint32_t max_seq, uint32_t num_heads) {
    for (uint32_t h = 0; h < num_heads; h++) {
        float* row = data + h * max_seq;
        float maxv = row[0];
        for (uint32_t p = 1; p < seq_len; p++) maxv = std::max(maxv, row[p]);
        float sum = 0;
        for (uint32_t p = 0; p < seq_len; p++) {
            row[p] = std::exp(row[p] - maxv);
            sum += row[p];
        }
        float inv = 1.0f / sum;
        for (uint32_t p = 0; p < seq_len; p++) row[p] *= inv;
        for (uint32_t p = seq_len; p < max_seq; p++) row[p] = 0;
    }
}

inline void ref_attention_value(const float* scores, const float* V_cache,
                                 float* output, uint32_t head_dim,
                                 uint32_t seq_len, uint32_t max_seq,
                                 uint32_t num_q_heads, uint32_t gqa_ratio) {
    for (uint32_t h = 0; h < num_q_heads; h++) {
        uint32_t kv_h = h / gqa_ratio;
        for (uint32_t d = 0; d < head_dim; d++) {
            float sum = 0;
            for (uint32_t p = 0; p < seq_len; p++)
                sum += scores[h * max_seq + p] * V_cache[kv_h * max_seq * head_dim + p * head_dim + d];
            output[h * head_dim + d] = sum;
        }
    }
}

inline void ref_add(const float* a, const float* b, float* c, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) c[i] = a[i] + b[i];
}
