#pragma once

#include "animations.hpp"
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <array>
#include <random>
#include <variant>
#include <memory>
#include <numeric>
#include <complex>
#include <deque>
#include <vector>
#include <bitset>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef DEG2RAD
#define DEG2RAD(x) ((x) * 0.017453292519943295769f)
#endif

#ifndef RAD2DEG
#define RAD2DEG(x) ((x) * 57.295779513082320876f)
#endif

#ifndef INLINE
#if defined(__clang__) || defined(__GNUC__)
#define INLINE __attribute__((always_inline)) inline
#else
#define INLINE inline
#endif
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#define HAS_AVX512
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define HAS_NEON
#endif

#ifndef FAST_SINCOSF_DECLARED
#define FAST_SINCOSF_DECLARED
static inline void fast_sincosf(float rad, float& s, float& c) noexcept {
    s = std::sinf(rad);
    c = std::cosf(rad);
}
#endif

constexpr int CACHE_SIZE = 2;
constexpr int YAW_CACHE_SIZE = 16;
constexpr int MAX_TICKS = 3;
constexpr float EPSILON = 1e-4f;
constexpr int DELTA_WINDOW = 32;
constexpr int TABLE_SIZE = 361;
constexpr int STFT_SIZE = 64;
constexpr int STFT_HOP = 16;

template<typename T, size_t N>
struct alignas(64) ring_buffer {
    alignas(64) T data[N]{};
    size_t index{};
    size_t count{};

    INLINE void push(const T& value) {
        data[index] = value;
        index = (index + 1) % N;
        count = std::min(count + 1, N);
    }

    INLINE T& operator[](size_t i) { return data[(index + N - 1 - i) % N]; }
    INLINE const T& operator[](size_t i) const { return data[(index + N - 1 - i) % N]; }

    INLINE T& get_chronological(size_t i) {
        return data[(index + N - 1 - i) % N];
    }

    INLINE const T& get_chronological(size_t i) const {
        return data[(index + N - 1 - i) % N];
    }

    INLINE size_t size() const { return count; }
};

struct alignas(64) trig_tables {
    static inline std::array<float, TABLE_SIZE> sin_table{};
    static inline std::array<float, TABLE_SIZE> cos_table{};

    struct initializer {
        initializer() {
            for (int i = 0; i < TABLE_SIZE; ++i) {
                sin_table[i] = std::sinf(DEG2RAD(float(i)));
                cos_table[i] = std::cosf(DEG2RAD(float(i)));
            }
        }
    };
    static inline initializer init{};
};

struct resolver_context {
    float latency{};
    int tick_rate{};
    int choked_commands{};
    bool is_competitive{};

    std::array<float, 4> make_feature_vector() const {
        return {
            latency * 0.01f,
            float(tick_rate) / 128.f,
            float(choked_commands) * 0.1f,
            is_competitive ? 1.f : 0.f
        };
    }
};

struct square_root_ukf_t {
    static constexpr int N = 5;
    static constexpr int L = 2 * N + 1;

    std::array<float, N> x{};
    std::array<std::array<float, N>, N> S{};
    std::array<float, N> Q_sqrt{ 1.f, 0.316f, 0.224f, 0.316f, 0.316f };
    std::array<float, 2> R_sqrt{ 2.f, 2.f };

    std::array<std::array<float, N>, L> sigma_points{};
    std::array<float, L> weights_mean{};
    std::array<float, L> weights_cov{};

    float alpha{ 0.3f };
    float beta{ 2.f };
    float kappa{ 0.f };
    float dt{ 1.f / 64.f };

    INLINE void init() {
        float lambda = alpha * alpha * (N + kappa) - N;
        weights_mean[0] = lambda / (N + lambda);
        weights_cov[0] = weights_mean[0] + (1 - alpha * alpha + beta);

        for (int i = 1; i < L; ++i) {
            weights_mean[i] = weights_cov[i] = 0.5f / (N + lambda);
        }

        for (int i = 0; i < N; ++i) {
            S[i][i] = 5.f;
        }
    }

    INLINE void set_dt(float delta_time) {
        dt = delta_time;
    }

    INLINE void generate_sigma_points() {
        std::copy(x.begin(), x.end(), sigma_points[0].begin());

        float lambda = alpha * alpha * (N + kappa) - N;
        float scale = std::sqrt(N + lambda);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                sigma_points[i + 1][j] = x[j] + scale * S[j][i];
                sigma_points[i + 1 + N][j] = x[j] - scale * S[j][i];
            }
        }
    }

    INLINE void predict() {
        generate_sigma_points();

        for (auto& sp : sigma_points) {
            sp[0] = wrap_deg(sp[0] + dt * sp[1] + 0.5f * dt * dt * sp[2]);
            sp[1] += dt * sp[2];

            float rad = DEG2RAD(sp[0]);
            fast_sincosf(rad, sp[3], sp[4]);
        }

        std::fill(x.begin(), x.end(), 0.f);
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < N; ++j) {
                x[j] += weights_mean[i] * sigma_points[i][j];
            }
        }
        x[0] = wrap_deg(x[0]);

        qr_update();
    }

    INLINE void update(float meas_yaw_deg) {
        float sin_m, cos_m;
        fast_sincosf(DEG2RAD(meas_yaw_deg), sin_m, cos_m);

        std::array<float, 2> z{ sin_m, cos_m };
        std::array<float, 2> z_pred{};

        for (int i = 0; i < L; ++i) {
            z_pred[0] += weights_mean[i] * sigma_points[i][3];
            z_pred[1] += weights_mean[i] * sigma_points[i][4];
        }

        std::array<std::array<float, 2>, L> Z{};
        std::array<std::array<float, N>, L> X{};

        for (int i = 0; i < L; ++i) {
            Z[i][0] = sigma_points[i][3] - z_pred[0];
            Z[i][1] = sigma_points[i][4] - z_pred[1];

            for (int j = 0; j < N; ++j) {
                X[i][j] = sigma_points[i][j] - x[j];
            }
            X[i][0] = wrap_deg(X[i][0]);
        }

        std::array<std::array<float, 2>, 2> Sz{};
        std::array<std::array<float, 2>, N> Pxz{};

        for (int i = 0; i < L; ++i) {
            float w = i == 0 ? weights_cov[0] : weights_cov[i];
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    Sz[j][k] += w * Z[i][j] * Z[i][k];
                }
            }

            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < 2; ++k) {
                    Pxz[j][k] += w * X[i][j] * Z[i][k];
                }
            }
        }

        Sz[0][0] += R_sqrt[0] * R_sqrt[0];
        Sz[1][1] += R_sqrt[1] * R_sqrt[1];

        std::array<std::array<float, 2>, 2> Sz_inv{};
        matrix_inverse_2x2(Sz, Sz_inv);

        std::array<std::array<float, 2>, N> K{};
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    K[i][j] += Pxz[i][k] * Sz_inv[k][j];
                }
            }
        }

        std::array<float, 2> y{ z[0] - z_pred[0], z[1] - z_pred[1] };

        for (int i = 0; i < N; ++i) {
            x[i] += K[i][0] * y[0] + K[i][1] * y[1];
        }
        x[0] = wrap_deg(x[0]);

        square_root_update(K, Sz);
    }

    INLINE float estimate() const { return x[0]; }

    INLINE float confidence() const noexcept {
        float sigma = std::fabs(S[0][0]);
        return std::clamp(1.f - sigma / 180.f, 0.f, 1.f);
    }

    INLINE void reset() {
        std::fill(x.begin(), x.end(), 0.f);
        for (auto& row : S) {
            std::fill(row.begin(), row.end(), 0.f);
        }
        for (int i = 0; i < N; ++i) {
            S[i][i] = 5.f;
        }
        init();
    }

private:
    static INLINE float wrap_deg(float a) {
        a = std::fmod(a + 180.f, 360.f);
        if (a < 0) a += 360.f;
        return a - 180.f;
    }

    void qr_update() {
        std::array<std::array<float, N>, L - 1 + N> B{};
        int col = 0;

        for (int i = 1; i < L; ++i, ++col) {
            float w = std::sqrt(std::abs(weights_cov[i]));
            for (int r = 0; r < N; ++r) {
                float v = sigma_points[i][r] - x[r];
                if (r == 0) {
                    if (v > 180.f) v -= 360.f;
                    if (v < -180.f) v += 360.f;
                }
                B[col][r] = w * v;
            }
        }

        for (int i = 0; i < N; ++i, ++col) {
            B[col][i] = Q_sqrt[i];
        }

        std::array<std::array<float, N>, N> R{};
        householder_QR_transpose(B, R);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                S[i][j] = (j <= i) ? R[j][i] : 0.f;
            }
        }

        {
            const float w0 = weights_cov[0];
            if (std::fabs(w0) > EPSILON) {
                std::array<float, N> u{};
                for (int i = 0; i < N; ++i) {
                    float v = sigma_points[0][i] - x[i];
                    if (i == 0) v = wrap_deg(v);
                    u[i] = std::sqrt(std::fabs(w0)) * v;
                }
                cholupdate(S, u, (w0 > 0.f) ? +1.f : -1.f);
            }
        }
    }

    void square_root_update(const std::array<std::array<float, 2>, N>& K,
        const std::array<std::array<float, 2>, 2>& Sz) {
        for (int k = 0; k < 2; ++k) {
            std::array<float, N> u{};
            float r = std::sqrt(Sz[k][k]);
            for (int i = 0; i < N; ++i) {
                u[i] = K[i][k] * r;
            }
            cholupdate(S, u, -1.f);
        }
    }

    static void matrix_inverse_2x2(const std::array<std::array<float, 2>, 2>& A,
        std::array<std::array<float, 2>, 2>& inv) {
        float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
        if (std::fabs(det) < EPSILON) {
            det = (det >= 0.f ? 1.f : -1.f) * EPSILON;
        }
        float inv_det = 1.f / det;
        inv[0][0] = A[1][1] * inv_det;
        inv[0][1] = -A[0][1] * inv_det;
        inv[1][0] = -A[1][0] * inv_det;
        inv[1][1] = A[0][0] * inv_det;
    }

    static void householder_QR_transpose(const std::array<std::array<float, N>, L - 1 + N>& B,
        std::array<std::array<float, N>, N>& R) {
        std::array<std::array<float, L - 1 + N>, N> A{};
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < L - 1 + N; ++j) {
                A[i][j] = B[j][i];
            }
        }

        for (int k = 0; k < N; ++k) {
            float norm = 0.f;
            for (int i = k; i < L - 1 + N; ++i) {
                norm += A[k][i] * A[k][i];
            }
            norm = std::sqrt(norm);

            if (norm > EPSILON) {
                if (A[k][k] < 0) norm = -norm;

                A[k][k] += norm;
                float beta = 1.f / (norm * A[k][k]);

                for (int j = k + 1; j < N; ++j) {
                    float dot = 0.f;
                    for (int i = k; i < L - 1 + N; ++i) {
                        dot += A[k][i] * A[j][i];
                    }
                    float tau = dot * beta;

                    for (int i = k; i < L - 1 + N; ++i) {
                        A[j][i] -= tau * A[k][i];
                    }
                }

                R[k][k] = std::fabs(norm);
                for (int j = k + 1; j < N; ++j) {
                    R[k][j] = A[j][k];
                }
            }
        }
    }

    static void cholupdate(std::array<std::array<float, N>, N>& L,
        const std::array<float, N>& u, float sign) {
        for (int k = 0; k < N; ++k) {
            float r = std::sqrt(L[k][k] * L[k][k] + sign * u[k] * u[k]);
            float c = r / L[k][k];
            float s = u[k] / L[k][k];

            L[k][k] = r;

            for (int i = k + 1; i < N; ++i) {
                L[i][k] = (L[i][k] + sign * s * u[i]) / c;
            }
        }
    }
};

struct ar1_particle_filter_t {
    static constexpr int NUM_PARTICLES = 128;
    static constexpr float DT = 1.0f / 64.0f;

    struct particle {
        float yaw{};
        float drift{};
        float weight{ 1.f / NUM_PARTICLES };
    };

    std::array<particle, NUM_PARTICLES> particles{};
    std::mt19937 rng{ std::random_device{}() };
    std::normal_distribution<float> drift_noise{ 0.f, 2.f };
    std::normal_distribution<float> measurement_noise{ 0.f, 10.f };

    float ar_coefficient{ 0.85f };
    float last_confidence{ 0.f };

    INLINE void init(float initial_yaw) {
        rng.seed(std::random_device{}());
        std::uniform_real_distribution<float> yaw_dist(-30.f, 30.f);
        std::uniform_real_distribution<float> drift_dist(-5.f, 5.f);

        for (auto& p : particles) {
            p.yaw = initial_yaw + yaw_dist(rng);
            p.drift = drift_dist(rng);
            p.weight = 1.f / NUM_PARTICLES;
        }
    }

    INLINE void predict() {
        std::normal_distribution<float> heading_noise{ 0.f, 0.5f };

        for (auto& p : particles) {
            p.drift = ar_coefficient * p.drift + drift_noise(rng);
            p.yaw = wrap_deg(p.yaw + p.drift * DT + heading_noise(rng));
        }
    }

    INLINE void update(float measured_yaw) {
        float weight_sum = 0.f;

        for (auto& p : particles) {
            float diff = signed_angle_diff(measured_yaw, p.yaw);
            p.weight = std::exp(-0.5f * diff * diff / (measurement_noise.stddev() * measurement_noise.stddev()));
            weight_sum += p.weight;
        }

        if (weight_sum > EPSILON) {
            for (auto& p : particles) {
                p.weight /= weight_sum;
            }
        }

        float neff_inv = 0.f;
        for (auto& p : particles) neff_inv += p.weight * p.weight;
        last_confidence = 1.f / neff_inv;
        last_confidence = std::clamp(last_confidence / NUM_PARTICLES, 0.f, 1.f);

        systematic_resample();
    }

    INLINE float estimate() const {
        float s = 0.f, c = 0.f;
        for (const auto& p : particles) {
            float rad = DEG2RAD(p.yaw);
            s += std::sin(rad) * p.weight;
            c += std::cos(rad) * p.weight;
        }
        return RAD2DEG(std::atan2(s, c));
    }

    INLINE float confidence() const noexcept { return last_confidence; }

private:
    static INLINE float signed_angle_diff(float a, float b) noexcept {
        float d = a - b;
        while (d > 180.f) d -= 360.f;
        while (d < -180.f) d += 360.f;
        return d;
    }

    INLINE void systematic_resample() {
        std::array<particle, NUM_PARTICLES> new_particles{};
        std::uniform_real_distribution<float> uniform(0.f, 1.f / NUM_PARTICLES);

        float start = uniform(rng);
        float cumsum = particles[0].weight;
        int j = 0;

        for (int i = 0; i < NUM_PARTICLES; ++i) {
            float u = start + float(i) / NUM_PARTICLES;

            while (cumsum < u && j < NUM_PARTICLES - 1) {
                ++j;
                cumsum += particles[j].weight;
            }

            new_particles[i] = particles[j];
            new_particles[i].weight = 1.f / NUM_PARTICLES;
        }

        particles = new_particles;
    }

    static INLINE float wrap_deg(float a) {
        a = std::fmod(a + 180.f, 360.f);
        if (a < 0) a += 360.f;
        return a - 180.f;
    }
};

struct ttt_resolver_t {
    static constexpr int INPUT_SIZE = 16;
    static constexpr int FEATURE_DIM = 64;
    static constexpr int HIDDEN_DIM = 128;
    static constexpr int OUTPUT_SIZE = 2;
    static constexpr int ADAPTATION_STEPS = 3;
    static constexpr int BUFFER_SIZE = 32;
    static constexpr int NUM_AUGMENTATIONS = 4;
    static constexpr float GRADIENT_CLIP = 5.0f;

    struct layer_t {
        std::array<std::array<float, FEATURE_DIM>, INPUT_SIZE> W1{};
        std::array<float, FEATURE_DIM> b1{};
        std::array<std::array<float, HIDDEN_DIM>, FEATURE_DIM> W2{};
        std::array<float, HIDDEN_DIM> b2{};
        std::array<std::array<float, HIDDEN_DIM>, HIDDEN_DIM> W3{};
        std::array<float, HIDDEN_DIM> b3{};
        std::array<std::array<float, OUTPUT_SIZE>, HIDDEN_DIM> W4{};
        std::array<float, OUTPUT_SIZE> b4{};

        std::array<std::array<float, FEATURE_DIM>, FEATURE_DIM> self_attn_q{};
        std::array<std::array<float, FEATURE_DIM>, FEATURE_DIM> self_attn_k{};
        std::array<std::array<float, FEATURE_DIM>, FEATURE_DIM> self_attn_v{};

        std::array<std::array<float, HIDDEN_DIM>, HIDDEN_DIM> adaptation_W{};
        std::array<float, HIDDEN_DIM> adaptation_b{};
    };

    struct forward_cache_t {
        std::array<float, INPUT_SIZE> norm_input{};
        std::array<float, FEATURE_DIM> features{};
        std::array<float, FEATURE_DIM> attn_output{};
        std::array<float, HIDDEN_DIM> h1{};
        std::array<float, HIDDEN_DIM> h2{};
        std::array<float, HIDDEN_DIM> adapted{};
        std::bitset<FEATURE_DIM> dropout_mask{};
        float attn_score{};
        float attn_weight{};
    };

    layer_t main_model{};
    layer_t adaptation_model{};

    struct momentum_buffer_t {
        std::array<std::array<float, FEATURE_DIM>, INPUT_SIZE> W1_m{}, W1_v{};
        std::array<float, FEATURE_DIM> b1_m{}, b1_v{};
        std::array<std::array<float, HIDDEN_DIM>, FEATURE_DIM> W2_m{}, W2_v{};
        std::array<float, HIDDEN_DIM> b2_m{}, b2_v{};
        std::array<std::array<float, HIDDEN_DIM>, HIDDEN_DIM> W3_m{}, W3_v{};
        std::array<float, HIDDEN_DIM> b3_m{}, b3_v{};
        std::array<std::array<float, OUTPUT_SIZE>, HIDDEN_DIM> W4_m{}, W4_v{};
        std::array<float, OUTPUT_SIZE> b4_m{}, b4_v{};

        std::array<std::array<float, FEATURE_DIM>, FEATURE_DIM> attn_q_m{}, attn_q_v{};
        std::array<std::array<float, FEATURE_DIM>, FEATURE_DIM> attn_k_m{}, attn_k_v{};
        std::array<std::array<float, FEATURE_DIM>, FEATURE_DIM> attn_v_m{}, attn_v_v{};

        std::array<std::array<float, HIDDEN_DIM>, HIDDEN_DIM> adapt_W_m{}, adapt_W_v{};
        std::array<float, HIDDEN_DIM> adapt_b_m{}, adapt_b_v{};
    } momentum{};

    float learning_rate{ 0.001f };
    float adaptation_lr{ 0.01f };
    float backbone_lr{ 0.0001f };
    float momentum_decay{ 0.9f };
    float weight_decay{ 0.0001f };
    float temperature{ 0.07f };
    float dropout_rate{ 0.1f };

    int adam_t{ 0 };
    float beta1{ 0.9f };
    float beta2{ 0.999f };
    float adam_epsilon{ 1e-8f };

    std::mt19937 rng{ std::random_device{}() };

    std::deque<std::pair<std::array<float, INPUT_SIZE>, std::pair<int, bool>>> replay_buffer;

    std::array<float, INPUT_SIZE> input_mean{};
    std::array<float, INPUT_SIZE> input_std{};
    bool stats_initialized{ false };

    struct augmentation_t {
        std::array<float, INPUT_SIZE> noise{};
        std::array<float, INPUT_SIZE> scale{};
        std::array<float, INPUT_SIZE> shift{};

        void generate(std::mt19937& rng) {
            std::normal_distribution<float> noise_dist(0.f, 0.05f);
            std::uniform_real_distribution<float> scale_dist(0.9f, 1.1f);
            std::uniform_real_distribution<float> shift_dist(-0.1f, 0.1f);

            for (int i = 0; i < INPUT_SIZE; ++i) {
                noise[i] = noise_dist(rng);
                scale[i] = scale_dist(rng);
                shift[i] = shift_dist(rng);
            }
        }

        std::array<float, INPUT_SIZE> apply(const std::array<float, INPUT_SIZE>& input) const {
            std::array<float, INPUT_SIZE> augmented{};
            for (int i = 0; i < INPUT_SIZE; ++i) {
                augmented[i] = input[i] * scale[i] + shift[i] + noise[i];
            }
            return augmented;
        }
    };

    INLINE void init() {
        std::normal_distribution<float> xavier(0.f, 1.f);

        auto init_layer = [&](layer_t& layer) {
            float scale1 = std::sqrt(2.f / INPUT_SIZE);
            for (auto& row : layer.W1) {
                for (auto& w : row) {
                    w = xavier(rng) * scale1;
                }
            }

            float scale2 = std::sqrt(2.f / FEATURE_DIM);
            for (auto& row : layer.W2) {
                for (auto& w : row) {
                    w = xavier(rng) * scale2;
                }
            }

            float scale3 = std::sqrt(2.f / HIDDEN_DIM);
            for (auto& row : layer.W3) {
                for (auto& w : row) {
                    w = xavier(rng) * scale3;
                }
            }

            float scale4 = std::sqrt(2.f / HIDDEN_DIM);
            for (auto& row : layer.W4) {
                for (auto& w : row) {
                    w = xavier(rng) * scale4;
                }
            }

            float attn_scale = std::sqrt(1.f / FEATURE_DIM);
            for (auto& row : layer.self_attn_q) {
                for (auto& w : row) {
                    w = xavier(rng) * attn_scale;
                }
            }
            for (auto& row : layer.self_attn_k) {
                for (auto& w : row) {
                    w = xavier(rng) * attn_scale;
                }
            }
            for (auto& row : layer.self_attn_v) {
                for (auto& w : row) {
                    w = xavier(rng) * attn_scale;
                }
            }

            for (int i = 0; i < HIDDEN_DIM; ++i) {
                layer.adaptation_W[i][i] = 1.f;
            }
            };

        init_layer(main_model);
        init_layer(adaptation_model);

        momentum = {};
    }

    INLINE void set_learning_rate(float lr) { learning_rate = lr; }
    INLINE void set_adaptation_lr(float lr) { adaptation_lr = lr; }
    INLINE void set_backbone_lr(float lr) { backbone_lr = lr; }
    INLINE void set_weight_decay(float wd) { weight_decay = wd; }

    INLINE void update_input_stats(const std::array<float, INPUT_SIZE>& input) {
        if (!stats_initialized) {
            input_mean = input;
            std::fill(input_std.begin(), input_std.end(), 1.f);
            stats_initialized = true;
            return;
        }

        float alpha = 0.01f;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            float delta = input[i] - input_mean[i];
            input_mean[i] += alpha * delta;
            float delta2 = input[i] - input_mean[i];
            float new_var = (1.f - alpha) * (input_std[i] * input_std[i]) + alpha * (delta * delta2);
            input_std[i] = std::sqrt(std::max(new_var, 1e-6f));
        }
    }

    INLINE std::array<float, INPUT_SIZE> normalize_input(const std::array<float, INPUT_SIZE>& input) {
        std::array<float, INPUT_SIZE> normalized{};
        for (int i = 0; i < INPUT_SIZE; ++i) {
            normalized[i] = (input[i] - input_mean[i]) / (input_std[i] + 1e-6f);
        }
        return normalized;
    }

    INLINE std::array<float, OUTPUT_SIZE> forward_with_model(layer_t& model,
        const std::array<float, INPUT_SIZE>& input,
        bool training = false,
        forward_cache_t* cache = nullptr) {
        forward_cache_t temp_cache;
        forward_cache_t& c = cache ? *cache : temp_cache;

        c.norm_input = normalize_input(input);

        for (int j = 0; j < FEATURE_DIM; ++j) {
            float sum = model.b1[j];
            for (int i = 0; i < INPUT_SIZE; ++i) {
                sum += model.W1[i][j] * c.norm_input[i];
            }
            c.features[j] = std::max(0.f, sum);
        }

        std::array<std::array<float, FEATURE_DIM>, 1> Q{}, K{}, V{};

        for (int i = 0; i < FEATURE_DIM; ++i) {
            Q[0][i] = 0.f;
            K[0][i] = 0.f;
            V[0][i] = 0.f;
            for (int j = 0; j < FEATURE_DIM; ++j) {
                Q[0][i] += model.self_attn_q[j][i] * c.features[j];
                K[0][i] += model.self_attn_k[j][i] * c.features[j];
                V[0][i] += model.self_attn_v[j][i] * c.features[j];
            }
        }

        float scale = 1.f / std::sqrt(float(FEATURE_DIM));
        c.attn_score = 0.f;
        for (int i = 0; i < FEATURE_DIM; ++i) {
            c.attn_score += Q[0][i] * K[0][i];
        }
        c.attn_score *= scale;

        c.attn_weight = 1.f / (1.f + std::exp(-c.attn_score));

        for (int i = 0; i < FEATURE_DIM; ++i) {
            c.attn_output[i] = c.features[i] + c.attn_weight * V[0][i];
        }

        std::uniform_real_distribution<float> dropout_dist(0.f, 1.f);
        if (training) {
            for (int i = 0; i < FEATURE_DIM; ++i) {
                bool keep = (dropout_dist(rng) >= dropout_rate);
                c.dropout_mask[i] = keep;
                if (!keep) {
                    c.attn_output[i] = 0.f;
                }
                else {
                    c.attn_output[i] *= 1.f / (1.f - dropout_rate);
                }
            }
        }

        for (int j = 0; j < HIDDEN_DIM; ++j) {
            float sum = model.b2[j];
            for (int i = 0; i < FEATURE_DIM; ++i) {
                sum += model.W2[i][j] * c.attn_output[i];
            }
            c.h1[j] = std::max(0.f, sum);
        }

        for (int j = 0; j < HIDDEN_DIM; ++j) {
            float sum = model.b3[j];
            for (int i = 0; i < HIDDEN_DIM; ++i) {
                sum += model.W3[i][j] * c.h1[i];
            }
            c.h2[j] = std::max(0.f, sum);
        }

        for (int j = 0; j < HIDDEN_DIM; ++j) {
            float sum = model.adaptation_b[j];
            for (int i = 0; i < HIDDEN_DIM; ++i) {
                sum += model.adaptation_W[i][j] * c.h2[i];
            }
            c.adapted[j] = sum;
        }

        std::array<float, OUTPUT_SIZE> output{};
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            float sum = model.b4[j];
            for (int i = 0; i < HIDDEN_DIM; ++i) {
                sum += model.W4[i][j] * c.adapted[i];
            }
            output[j] = sum;
        }

        float max_val = *std::max_element(output.begin(), output.end());
        float sum_exp = 0.f;

        for (auto& o : output) {
            o = std::exp(o - max_val);
            sum_exp += o;
        }

        for (auto& o : output) {
            o /= sum_exp;
        }

        return output;
    }

    INLINE std::array<float, OUTPUT_SIZE> forward(const std::array<float, INPUT_SIZE>& input, bool training = false) {
        return forward_with_model(adaptation_model, input, training);
    }

    INLINE void test_time_adapt(const std::array<float, INPUT_SIZE>& input) {
        adaptation_model = main_model;

        augmentation_t augmenter;

        for (int step = 0; step < ADAPTATION_STEPS; ++step) {
            std::array<std::array<float, FEATURE_DIM>, NUM_AUGMENTATIONS> aug_features{};
            std::array<std::array<float, HIDDEN_DIM>, NUM_AUGMENTATIONS> h2_cache{};
            std::array<forward_cache_t, NUM_AUGMENTATIONS> forward_caches{};

            for (int aug = 0; aug < NUM_AUGMENTATIONS; ++aug) {
                augmenter.generate(rng);
                auto augmented = augmenter.apply(input);
                forward_with_model(adaptation_model, augmented, true, &forward_caches[aug]);
                aug_features[aug] = forward_caches[aug].attn_output;
                h2_cache[aug] = forward_caches[aug].h2;
            }

            std::array<std::array<float, NUM_AUGMENTATIONS>, NUM_AUGMENTATIONS> sims{};
            for (int i = 0; i < NUM_AUGMENTATIONS; ++i) {
                for (int j = 0; j < NUM_AUGMENTATIONS; ++j) {
                    float sim = 0.f;
                    float norm_i = 0.f, norm_j = 0.f;

                    for (int k = 0; k < FEATURE_DIM; ++k) {
                        sim += aug_features[i][k] * aug_features[j][k];
                        norm_i += aug_features[i][k] * aug_features[i][k];
                        norm_j += aug_features[j][k] * aug_features[j][k];
                    }

                    float denom = std::sqrt(std::max(norm_i, 1e-12f)) * std::sqrt(std::max(norm_j, 1e-12f));
                    sims[i][j] = sim / denom;
                }
            }

            float contrastive_loss = 0.f;
            std::array<std::array<float, FEATURE_DIM>, NUM_AUGMENTATIONS> grad_features{};

            for (int a = 0; a < NUM_AUGMENTATIONS; ++a) {
                int p = (a + 1) % NUM_AUGMENTATIONS;
                float numerator = std::exp(sims[a][p] / temperature);
                float denominator = 0.f;

                for (int k = 0; k < NUM_AUGMENTATIONS; ++k) {
                    if (k != a) {
                        denominator += std::exp(sims[a][k] / temperature);
                    }
                }

                float prob = numerator / std::max(denominator, 1e-12f);
                contrastive_loss += -std::log(std::max(prob, 1e-12f));

                for (int k = 0; k < NUM_AUGMENTATIONS; ++k) {
                    if (k != a) {
                        float exp_sim = std::exp(sims[a][k] / temperature);
                        float grad_scale = exp_sim / std::max(denominator, 1e-12f) / temperature;

                        if (k == p) {
                            grad_scale -= 1.f / temperature;
                        }

                        for (int feat = 0; feat < FEATURE_DIM; ++feat) {
                            grad_features[a][feat] += grad_scale * aug_features[k][feat];
                        }
                    }
                }
            }

            contrastive_loss /= NUM_AUGMENTATIONS;

            gradient_step_adaptation(h2_cache, grad_features, forward_caches);
        }
    }

    INLINE void gradient_step_adaptation(const std::array<std::array<float, HIDDEN_DIM>, NUM_AUGMENTATIONS>& h2_cache,
        const std::array<std::array<float, FEATURE_DIM>, NUM_AUGMENTATIONS>& grad_features,
        const std::array<forward_cache_t, NUM_AUGMENTATIONS>& caches) {
        auto& model = adaptation_model;

        for (int i = 0; i < HIDDEN_DIM; ++i) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                float grad = 0.f;
                model.adaptation_W[i][j] -= adaptation_lr * (grad + weight_decay * model.adaptation_W[i][j]);
            }
        }

        for (int i = 0; i < HIDDEN_DIM; ++i) {
            float grad = 0.f;
            model.adaptation_b[i] -= adaptation_lr * grad;
        }

        for (int aug = 0; aug < NUM_AUGMENTATIONS; ++aug) {
            const auto& cache = caches[aug];
            const auto& grad_feat = grad_features[aug];

            std::array<float, HIDDEN_DIM> grad_h2{};
            std::array<float, HIDDEN_DIM> grad_h1{};
            std::array<float, FEATURE_DIM> grad_attn = grad_feat;

            for (int i = 0; i < FEATURE_DIM; ++i) {
                if (!cache.dropout_mask[i]) {
                    grad_attn[i] = 0.f;
                }
            }

            for (int i = 0; i < FEATURE_DIM; ++i) {
                for (int j = 0; j < HIDDEN_DIM; ++j) {
                    float grad = grad_attn[i] * cache.attn_output[i];
                    grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);
                    model.W2[i][j] -= backbone_lr * (grad + weight_decay * model.W2[i][j]);
                }
            }

            std::array<float, FEATURE_DIM> grad_V{};
            for (int i = 0; i < FEATURE_DIM; ++i) {
                grad_V[i] = grad_attn[i] * cache.attn_weight;
            }

            for (int i = 0; i < FEATURE_DIM; ++i) {
                for (int j = 0; j < FEATURE_DIM; ++j) {
                    float grad = grad_V[i] * cache.features[j];
                    grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);
                    model.self_attn_v[j][i] -= backbone_lr * (grad + weight_decay * model.self_attn_v[j][i]);
                }
            }

            float grad_attn_weight = 0.f;
            for (int i = 0; i < FEATURE_DIM; ++i) {
                grad_attn_weight += grad_attn[i] * grad_V[i];
            }

            float sigmoid_grad = cache.attn_weight * (1.f - cache.attn_weight);
            float grad_score = grad_attn_weight * sigmoid_grad;

            float scale = 1.f / std::sqrt(float(FEATURE_DIM));
            std::array<float, FEATURE_DIM> grad_Q{}, grad_K{};

            for (int i = 0; i < FEATURE_DIM; ++i) {
                grad_Q[i] = grad_score * scale * cache.features[i];
                grad_K[i] = grad_score * scale * cache.features[i];
            }

            for (int i = 0; i < FEATURE_DIM; ++i) {
                for (int j = 0; j < FEATURE_DIM; ++j) {
                    float grad_q = grad_Q[i] * cache.features[j];
                    float grad_k = grad_K[i] * cache.features[j];

                    grad_q = std::clamp(grad_q, -GRADIENT_CLIP, GRADIENT_CLIP);
                    grad_k = std::clamp(grad_k, -GRADIENT_CLIP, GRADIENT_CLIP);

                    model.self_attn_q[j][i] -= backbone_lr * (grad_q + weight_decay * model.self_attn_q[j][i]);
                    model.self_attn_k[j][i] -= backbone_lr * (grad_k + weight_decay * model.self_attn_k[j][i]);
                }
            }
        }
    }

    INLINE void update(const std::array<float, INPUT_SIZE>& input, int true_side, bool hit) {
        update_input_stats(input);

        replay_buffer.push_back({ input, {true_side, hit} });
        if (replay_buffer.size() > BUFFER_SIZE) {
            replay_buffer.pop_front();
        }

        if (replay_buffer.size() >= 8) {
            train_batch();
        }
    }

    INLINE void train_batch() {
        const int batch_size = std::min(16, static_cast<int>(replay_buffer.size()));

        for (int iter = 0; iter < 2; ++iter) {
            ++adam_t;

            std::vector<int> indices(replay_buffer.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);

            int effective_batch = 0;

            for (int b = 0; b < batch_size; ++b) {
                const auto& [input, target_data] = replay_buffer[indices[b]];
                const auto& [true_side, hit] = target_data;

                if (true_side == 0) {
                    continue;
                }

                effective_batch++;

                forward_cache_t cache;
                auto output = forward_with_model(main_model, input, true, &cache);

                std::array<float, OUTPUT_SIZE> grad_output{};
                float target0 = true_side < 0 ? 1.f : 0.f;
                float target1 = true_side > 0 ? 1.f : 0.f;

                grad_output[0] = (output[0] - target0) / effective_batch;
                grad_output[1] = (output[1] - target1) / effective_batch;

                backward_pass(grad_output, cache);
            }

            if (effective_batch == 0) {
                return;
            }
        }
    }

    INLINE void backward_pass(const std::array<float, OUTPUT_SIZE>& grad_output,
        const forward_cache_t& cache) {

        auto& model = main_model;
        auto& mom = momentum;

        std::array<float, HIDDEN_DIM> grad_adapted{};
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                grad_adapted[i] += grad_output[j] * model.W4[i][j];
            }
        }

        for (int i = 0; i < HIDDEN_DIM; ++i) {
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                float grad = grad_output[j] * cache.adapted[i];
                grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

                mom.W4_m[i][j] = beta1 * mom.W4_m[i][j] + (1.f - beta1) * grad;
                mom.W4_v[i][j] = beta2 * mom.W4_v[i][j] + (1.f - beta2) * grad * grad;

                float m_hat = mom.W4_m[i][j] / (1.f - std::pow(beta1, adam_t));
                float v_hat = mom.W4_v[i][j] / (1.f - std::pow(beta2, adam_t));

                model.W4[i][j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon) +
                    weight_decay * model.W4[i][j]);
            }
        }

        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            float grad = grad_output[j];
            grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

            mom.b4_m[j] = beta1 * mom.b4_m[j] + (1.f - beta1) * grad;
            mom.b4_v[j] = beta2 * mom.b4_v[j] + (1.f - beta2) * grad * grad;

            float m_hat = mom.b4_m[j] / (1.f - std::pow(beta1, adam_t));
            float v_hat = mom.b4_v[j] / (1.f - std::pow(beta2, adam_t));

            model.b4[j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon));
        }

        std::array<float, HIDDEN_DIM> grad_h2{};
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                grad_h2[i] += grad_adapted[j] * model.adaptation_W[i][j];
            }
            if (cache.h2[i] <= 0.f) grad_h2[i] = 0.f;
        }

        for (int i = 0; i < HIDDEN_DIM; ++i) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                float grad = grad_adapted[j] * cache.h2[i];
                grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

                mom.adapt_W_m[i][j] = beta1 * mom.adapt_W_m[i][j] + (1.f - beta1) * grad;
                mom.adapt_W_v[i][j] = beta2 * mom.adapt_W_v[i][j] + (1.f - beta2) * grad * grad;

                float m_hat = mom.adapt_W_m[i][j] / (1.f - std::pow(beta1, adam_t));
                float v_hat = mom.adapt_W_v[i][j] / (1.f - std::pow(beta2, adam_t));

                model.adaptation_W[i][j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon) +
                    weight_decay * model.adaptation_W[i][j]);
            }
        }

        for (int j = 0; j < HIDDEN_DIM; ++j) {
            float grad = grad_adapted[j];
            grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

            mom.adapt_b_m[j] = beta1 * mom.adapt_b_m[j] + (1.f - beta1) * grad;
            mom.adapt_b_v[j] = beta2 * mom.adapt_b_v[j] + (1.f - beta2) * grad * grad;

            float m_hat = mom.adapt_b_m[j] / (1.f - std::pow(beta1, adam_t));
            float v_hat = mom.adapt_b_v[j] / (1.f - std::pow(beta2, adam_t));

            model.adaptation_b[j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon));
        }

        std::array<float, HIDDEN_DIM> grad_h1{};
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                grad_h1[i] += grad_h2[j] * model.W3[i][j];
            }
            if (cache.h1[i] <= 0.f) grad_h1[i] = 0.f;
        }

        for (int i = 0; i < HIDDEN_DIM; ++i) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                float grad = grad_h2[j] * cache.h1[i];
                grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

                mom.W3_m[i][j] = beta1 * mom.W3_m[i][j] + (1.f - beta1) * grad;
                mom.W3_v[i][j] = beta2 * mom.W3_v[i][j] + (1.f - beta2) * grad * grad;

                float m_hat = mom.W3_m[i][j] / (1.f - std::pow(beta1, adam_t));
                float v_hat = mom.W3_v[i][j] / (1.f - std::pow(beta2, adam_t));

                model.W3[i][j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon) +
                    weight_decay * model.W3[i][j]);
            }
        }

        for (int j = 0; j < HIDDEN_DIM; ++j) {
            float grad = grad_h2[j];
            grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

            mom.b3_m[j] = beta1 * mom.b3_m[j] + (1.f - beta1) * grad;
            mom.b3_v[j] = beta2 * mom.b3_v[j] + (1.f - beta2) * grad * grad;

            float m_hat = mom.b3_m[j] / (1.f - std::pow(beta1, adam_t));
            float v_hat = mom.b3_v[j] / (1.f - std::pow(beta2, adam_t));

            model.b3[j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon));
        }

        std::array<float, FEATURE_DIM> grad_attn{};
        for (int i = 0; i < FEATURE_DIM; ++i) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                grad_attn[i] += grad_h1[j] * model.W2[i][j];
            }
            if (!cache.dropout_mask[i]) {
                grad_attn[i] = 0.f;
            }
        }

        for (int i = 0; i < FEATURE_DIM; ++i) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                float grad = grad_h1[j] * cache.attn_output[i];
                grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

                mom.W2_m[i][j] = beta1 * mom.W2_m[i][j] + (1.f - beta1) * grad;
                mom.W2_v[i][j] = beta2 * mom.W2_v[i][j] + (1.f - beta2) * grad * grad;

                float m_hat = mom.W2_m[i][j] / (1.f - std::pow(beta1, adam_t));
                float v_hat = mom.W2_v[i][j] / (1.f - std::pow(beta2, adam_t));

                model.W2[i][j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon) +
                    weight_decay * model.W2[i][j]);
            }
        }

        for (int j = 0; j < HIDDEN_DIM; ++j) {
            float grad = grad_h1[j];
            grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

            mom.b2_m[j] = beta1 * mom.b2_m[j] + (1.f - beta1) * grad;
            mom.b2_v[j] = beta2 * mom.b2_v[j] + (1.f - beta2) * grad * grad;

            float m_hat = mom.b2_m[j] / (1.f - std::pow(beta1, adam_t));
            float v_hat = mom.b2_v[j] / (1.f - std::pow(beta2, adam_t));

            model.b2[j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon));
        }

        std::array<float, FEATURE_DIM> grad_features = grad_attn;
        std::array<float, FEATURE_DIM> grad_V{};

        for (int i = 0; i < FEATURE_DIM; ++i) {
            grad_features[i] += grad_attn[i];
            grad_V[i] = grad_attn[i] * cache.attn_weight;
        }

        for (int i = 0; i < FEATURE_DIM; ++i) {
            for (int j = 0; j < FEATURE_DIM; ++j) {
                float grad = grad_V[i] * cache.features[j];
                grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

                mom.attn_v_m[j][i] = beta1 * mom.attn_v_m[j][i] + (1.f - beta1) * grad;
                mom.attn_v_v[j][i] = beta2 * mom.attn_v_v[j][i] + (1.f - beta2) * grad * grad;

                float m_hat = mom.attn_v_m[j][i] / (1.f - std::pow(beta1, adam_t));
                float v_hat = mom.attn_v_v[j][i] / (1.f - std::pow(beta2, adam_t));

                model.self_attn_v[j][i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon) +
                    weight_decay * model.self_attn_v[j][i]);
            }
        }

        float grad_attn_weight = 0.f;
        for (int i = 0; i < FEATURE_DIM; ++i) {
            grad_attn_weight += grad_attn[i] * grad_V[i];
        }

        float sigmoid_grad = cache.attn_weight * (1.f - cache.attn_weight);
        float grad_score = grad_attn_weight * sigmoid_grad;

        float scale = 1.f / std::sqrt(float(FEATURE_DIM));

        for (int i = 0; i < FEATURE_DIM; ++i) {
            for (int j = 0; j < FEATURE_DIM; ++j) {
                float grad_q = grad_score * scale * cache.features[i] * cache.features[j];
                float grad_k = grad_score * scale * cache.features[i] * cache.features[j];

                grad_q = std::clamp(grad_q, -GRADIENT_CLIP, GRADIENT_CLIP);
                grad_k = std::clamp(grad_k, -GRADIENT_CLIP, GRADIENT_CLIP);

                mom.attn_q_m[j][i] = beta1 * mom.attn_q_m[j][i] + (1.f - beta1) * grad_q;
                mom.attn_q_v[j][i] = beta2 * mom.attn_q_v[j][i] + (1.f - beta2) * grad_q * grad_q;

                float m_hat_q = mom.attn_q_m[j][i] / (1.f - std::pow(beta1, adam_t));
                float v_hat_q = mom.attn_q_v[j][i] / (1.f - std::pow(beta2, adam_t));

                model.self_attn_q[j][i] -= learning_rate * (m_hat_q / (std::sqrt(v_hat_q) + adam_epsilon) +
                    weight_decay * model.self_attn_q[j][i]);

                mom.attn_k_m[j][i] = beta1 * mom.attn_k_m[j][i] + (1.f - beta1) * grad_k;
                mom.attn_k_v[j][i] = beta2 * mom.attn_k_v[j][i] + (1.f - beta2) * grad_k * grad_k;

                float m_hat_k = mom.attn_k_m[j][i] / (1.f - std::pow(beta1, adam_t));
                float v_hat_k = mom.attn_k_v[j][i] / (1.f - std::pow(beta2, adam_t));

                model.self_attn_k[j][i] -= learning_rate * (m_hat_k / (std::sqrt(v_hat_k) + adam_epsilon) +
                    weight_decay * model.self_attn_k[j][i]);
            }
        }

        for (int i = 0; i < FEATURE_DIM; ++i) {
            if (cache.features[i] <= 0.f) grad_features[i] = 0.f;
        }

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < FEATURE_DIM; ++j) {
                float grad = grad_features[j] * cache.norm_input[i];
                grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

                mom.W1_m[i][j] = beta1 * mom.W1_m[i][j] + (1.f - beta1) * grad;
                mom.W1_v[i][j] = beta2 * mom.W1_v[i][j] + (1.f - beta2) * grad * grad;

                float m_hat = mom.W1_m[i][j] / (1.f - std::pow(beta1, adam_t));
                float v_hat = mom.W1_v[i][j] / (1.f - std::pow(beta2, adam_t));

                model.W1[i][j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon) +
                    weight_decay * model.W1[i][j]);
            }
        }

        for (int j = 0; j < FEATURE_DIM; ++j) {
            float grad = grad_features[j];
            grad = std::clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);

            mom.b1_m[j] = beta1 * mom.b1_m[j] + (1.f - beta1) * grad;
            mom.b1_v[j] = beta2 * mom.b1_v[j] + (1.f - beta2) * grad * grad;

            float m_hat = mom.b1_m[j] / (1.f - std::pow(beta1, adam_t));
            float v_hat = mom.b1_v[j] / (1.f - std::pow(beta2, adam_t));

            model.b1[j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + adam_epsilon));
        }
    }

    INLINE int predict(const std::array<float, INPUT_SIZE>& input) {
        test_time_adapt(input);
        auto output = forward(input, false);
        return output[1] > output[0] ? 1 : -1;
    }
};

enum class resolver_state : uint8_t {
    IDLE,
    ANALYZING,
    RESOLVED,
    BRUTE_FORCING
};

struct resolver_info_t {
    bool resolved{};
    int side{};
    int legit_ticks{};
    int fake_ticks{};

    resolver_state state{ resolver_state::IDLE };
    std::string mode{};

    INLINE void add_legit_ticks() {
        if (legit_ticks >= MAX_TICKS) {
            fake_ticks = 0;
            legit_ticks = MAX_TICKS;
        }
        else {
            ++legit_ticks;
        }
    }

    INLINE void add_fake_ticks() {
        if (fake_ticks >= MAX_TICKS) {
            legit_ticks = 0;
            fake_ticks = MAX_TICKS;
        }
        else {
            ++fake_ticks;
        }
    }

    INLINE bool is_legit() {
        return legit_ticks > fake_ticks;
    }

    c_animation_layers initial_layers[13]{};

    struct jitter_info_t {
        bool is_jitter{};
        bool high_freq_jitter{};
        int frequency{};

        ring_buffer<float, DELTA_WINDOW> delta_history{};
        ring_buffer<float, CACHE_SIZE> delta_cache{};
        ring_buffer<float, YAW_CACHE_SIZE> yaw_cache{};

        float variance{};
        float strength{};
        float autocorr{};
        float delta_variance{};

        int jitter_ticks{};
        int static_ticks{};

        float sum_x{};
        float sum_x2{};

        struct stft_analyzer {
            std::array<std::complex<float>, STFT_SIZE> spectrum{};
            std::array<float, STFT_SIZE> window{};

            stft_analyzer() {
                for (int i = 0; i < STFT_SIZE; ++i) {
                    window[i] = 0.5f - 0.5f * std::cosf(2.f * M_PI * i / (STFT_SIZE - 1));
                }
            }

            void analyze(const ring_buffer<float, DELTA_WINDOW>& data) {
                if (data.size() < STFT_SIZE) return;

                std::array<std::complex<float>, STFT_SIZE> windowed{};
                for (int i = 0; i < STFT_SIZE; ++i) {
                    windowed[i] = data.get_chronological(i) * window[i];
                }

                fft(windowed.data(), spectrum.data(), STFT_SIZE);
            }

            float get_power(int bin) const {
                return std::abs(spectrum[bin]);
            }

        private:
            void fft(std::complex<float>* x, std::complex<float>* X, int N) {
                if (N <= 1) {
                    if (N == 1) X[0] = x[0];
                    return;
                }

                std::vector<std::complex<float>> even(N / 2), odd(N / 2);
                std::vector<std::complex<float>> Even(N / 2), Odd(N / 2);

                for (int i = 0; i < N / 2; ++i) {
                    even[i] = x[2 * i];
                    odd[i] = x[2 * i + 1];
                }

                fft(even.data(), Even.data(), N / 2);
                fft(odd.data(), Odd.data(), N / 2);

                for (int k = 0; k < N / 2; ++k) {
                    float angle = -2.f * M_PI * k / N;
                    std::complex<float> t = std::polar(1.f, angle) * Odd[k];
                    X[k] = Even[k] + t;
                    X[k + N / 2] = Even[k] - t;
                }
            }
        } stft;

        struct biquad_filter {
            float b0{ 1.f }, b1{}, b2{};
            float a1{}, a2{};
            float z1{}, z2{};

            void set_peak(float freq, float q, float gain) {
                float omega = 2.f * M_PI * freq;
                float alpha = std::sinf(omega) / (2.f * q);
                float A = std::powf(10.f, gain / 40.f);

                float a0 = 1.f + alpha / A;
                b0 = (1.f + alpha * A) / a0;
                b1 = (-2.f * std::cosf(omega)) / a0;
                b2 = (1.f - alpha * A) / a0;
                a1 = b1;
                a2 = (1.f - alpha / A) / a0;
            }

            float process(float x) {
                float w = x - a1 * z1 - a2 * z2;
                float y = b0 * w + b1 * z1 + b2 * z2;
                z2 = z1;
                z1 = w;
                return y;
            }
        } biquad;

        enum class jitter_state : uint8_t {
            STATIC = 0,
            JITTER = 1
        };

        struct jitter_state_machine_t {
            jitter_state state{ jitter_state::STATIC };
            int ticks_static{};
            int ticks_jitter{};

            static constexpr int TRANSITION_THRESHOLD = 2;

            INLINE void reset() {
                state = jitter_state::STATIC;
                ticks_static = 0;
                ticks_jitter = 0;
            }

            INLINE void update(bool is_jitter_now, float variance) {
                const bool spike = variance > 20.f;

                switch (state) {
                case jitter_state::STATIC:
                    if (is_jitter_now || spike) {
                        ++ticks_jitter;
                        ticks_static = std::max(0, ticks_static - 1);
                        if (spike || ticks_jitter > TRANSITION_THRESHOLD) {
                            state = jitter_state::JITTER;
                        }
                    }
                    else {
                        ++ticks_static;
                        ticks_jitter = std::max(0, ticks_jitter - 1);
                    }
                    break;

                case jitter_state::JITTER:
                    if (is_jitter_now || spike) {
                        ++ticks_jitter;
                        ticks_static = std::max(0, ticks_static - 1);
                    }
                    else {
                        ++ticks_static;
                        ticks_jitter = std::max(0, ticks_jitter - 1);
                        if (ticks_static > TRANSITION_THRESHOLD) {
                            state = jitter_state::STATIC;
                        }
                    }
                    break;
                }
            }

            INLINE bool is_jitter() const {
                return state == jitter_state::JITTER;
            }
        } machine{};

        INLINE void reset() {
            is_jitter = false;
            high_freq_jitter = false;
            frequency = 0;
            variance = 0.f;
            autocorr = 0.f;
            jitter_ticks = 0;
            static_ticks = 0;
            delta_variance = 0.f;
            machine.reset();
            sum_x = 0.f;
            sum_x2 = 0.f;
            biquad = {};
            biquad.set_peak(0.05f, 0.7f, 10.f);
        }

        INLINE bool is_high_jitter() const {
            return variance > 15.f && autocorr < 0.f;
        }
    } jitter;

    std::unique_ptr<square_root_ukf_t> ukf{ std::make_unique<square_root_ukf_t>() };
    std::unique_ptr<ar1_particle_filter_t> particle_filter{ std::make_unique<ar1_particle_filter_t>() };
    std::unique_ptr<ttt_resolver_t> ttt_net{ std::make_unique<ttt_resolver_t>() };

    struct thompson_sampling_bandit_t {
        static constexpr int NUM_ARMS = 7;
        static constexpr int CONTEXT_DIM = 8;

        struct arm_model_t {
            std::array<float, CONTEXT_DIM> mu{};
            std::array<std::array<float, CONTEXT_DIM>, CONTEXT_DIM> sigma{};
            float a{ 1.f };
            float b{ 1.f };

            INLINE void init() {
                std::fill(mu.begin(), mu.end(), 0.f);
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    sigma[i][i] = 1.f;
                }
            }

            INLINE float sample(const std::array<float, CONTEXT_DIM>& context, std::mt19937& rng) const {
                std::normal_distribution<float> normal(0.f, 1.f);

                std::array<float, CONTEXT_DIM> z{};
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    z[i] = normal(rng);
                }

                std::array<float, CONTEXT_DIM> sampled_theta{};
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    sampled_theta[i] = mu[i];
                    for (int j = 0; j < CONTEXT_DIM; ++j) {
                        sampled_theta[i] += sigma[i][j] * z[j];
                    }
                }

                float linear_pred = 0.f;
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    linear_pred += sampled_theta[i] * context[i];
                }

                std::gamma_distribution<float> gamma_a(a, 1.f);
                std::gamma_distribution<float> gamma_b(b, 1.f);
                float precision = gamma_a(rng) / (gamma_b(rng) + 1e-6f);

                return linear_pred + normal(rng) / std::sqrt(precision + 1e-6f);
            }

            INLINE void update(const std::array<float, CONTEXT_DIM>& context, float reward) {
                std::array<float, CONTEXT_DIM> sigma_x{};
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    for (int j = 0; j < CONTEXT_DIM; ++j) {
                        sigma_x[i] += sigma[i][j] * context[j];
                    }
                }

                float x_sigma_x = 0.f;
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    x_sigma_x += context[i] * sigma_x[i];
                }

                float k_denom = 1.f + x_sigma_x;
                std::array<float, CONTEXT_DIM> k{};
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    k[i] = sigma_x[i] / k_denom;
                }

                float pred = 0.f;
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    pred += mu[i] * context[i];
                }

                float error = reward - pred;
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    mu[i] += k[i] * error;
                }

                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    for (int j = 0; j < CONTEXT_DIM; ++j) {
                        sigma[i][j] -= k[i] * sigma_x[j];
                    }
                }

                a += 0.5f;
                b += 0.5f * error * error;
            }
        };

        std::array<arm_model_t, NUM_ARMS> models{};
        std::mt19937 rng{ std::random_device{}() };

        INLINE void init() {
            for (auto& model : models) {
                model.init();
            }
        }

        INLINE int select(const std::array<float, CONTEXT_DIM>& context) {
            int best_arm = 0;
            float best_value = models[0].sample(context, rng);

            for (int i = 1; i < NUM_ARMS; ++i) {
                float value = models[i].sample(context, rng);
                if (value > best_value) {
                    best_value = value;
                    best_arm = i;
                }
            }

            return best_arm;
        }

        INLINE void update(int arm, const std::array<float, CONTEXT_DIM>& context, bool hit) {
            float reward = hit ? 1.f : -0.5f;
            models[arm].update(context, reward);
        }
    } bandit;

    struct adaptive_confidence_t {
        std::array<float, 5> weights{ 0.2f, 0.2f, 0.2f, 0.2f, 0.2f };
        std::array<float, 5> performance{ 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
        float learning_rate{ 0.01f };
        float decay_rate{ 0.10f };
        float last_update{};

        INLINE float blend(const std::array<float, 5>& confidences) {
            float sum = 0.f;
            float weight_sum = 0.f;

            for (int i = 0; i < 5; ++i) {
                sum += confidences[i] * weights[i];
                weight_sum += weights[i];
            }

            float now = HACKS->global_vars->curtime;
            float age = now - last_update;
            float raw = weight_sum > EPSILON ? sum / weight_sum : 0.5f;
            return std::clamp(raw - decay_rate * age, 0.f, 1.f);
        }

        INLINE void update(int method_idx, bool success) {
            if (method_idx < 0 || method_idx >= 5) return;

            float target = success ? 1.f : 0.f;
            performance[method_idx] += learning_rate * (target - performance[method_idx]);
            performance[method_idx] = std::clamp(performance[method_idx], 0.1f, 0.9f);

            last_update = HACKS->global_vars->curtime;

            float total_perf = std::accumulate(performance.begin(), performance.end(), 0.f);
            if (total_perf > EPSILON) {
                for (int i = 0; i < 5; ++i) {
                    weights[i] = performance[i] / total_perf;
                }
            }
        }
    } adaptive_confidence;

    struct skeletal_analyzer_t {
        struct bone_data_t {
            vec3_t head_pos{};
            vec3_t chest_pos{};
            vec3_t pelvis_pos{};
            float spine_angle{};
            float lean_angle{};
        };

        bone_data_t current{};
        bone_data_t previous{};

        INLINE void update(c_cs_player* player) {
            previous = current;

            current.head_pos = player->get_hitbox_position(HITBOX_HEAD, nullptr);
            current.chest_pos = player->get_hitbox_position(HITBOX_CHEST, nullptr);
            current.pelvis_pos = player->get_hitbox_position(HITBOX_PELVIS, nullptr);

            vec3_t spine_dir = current.chest_pos - current.pelvis_pos;
            float spine_len = spine_dir.length();
            if (spine_len > EPSILON) {
                spine_dir = spine_dir / spine_len;
                current.spine_angle = RAD2DEG(std::atan2f(spine_dir.y, spine_dir.x));

                vec3_t vertical(0.f, 0.f, 1.f);
                float dot = spine_dir.dot(vertical);
                current.lean_angle = RAD2DEG(std::acosf(std::clamp(dot, -1.f, 1.f)));
            }
        }

        INLINE int predict_side() const {
            if (std::fabs(current.lean_angle) > 15.f) {
                return current.lean_angle > 0.f ? 1 : -1;
            }

            float spine_delta = math::normalize_yaw(current.spine_angle - previous.spine_angle);
            if (std::fabs(spine_delta) > 5.f) {
                return spine_delta > 0.f ? 1 : -1;
            }

            return 0;
        }
    } skeletal_analyzer;

    struct latency_compensator_t {
        float estimated_latency{};
        float jitter_compensation{};

        INLINE void update(const resolver_context& ctx) {
            estimated_latency = ctx.latency;

            float tick_interval = 1.f / float(ctx.tick_rate);
            jitter_compensation = tick_interval * float(ctx.choked_commands);
        }

        INLINE float compensate_angle(float angle, float angular_velocity) const {
            float total_delay = estimated_latency + jitter_compensation;
            return angle + angular_velocity * total_delay;
        }
    } latency_compensator;

    std::array<float, 7> brute_offsets{};
    std::array<int, 7> brute_hits{};
    int brute_step{};

    float max_desync{};
    int locked_side{};
    float lock_time{};

    struct side_confidence_t {
        float jitter{};
        float foot_delta{};
        float freestand{};
        float velocity{};
        float brute{};

        INLINE void reset() {
            jitter = 0.f;
            foot_delta = 0.f;
            freestand = 0.f;
            velocity = 0.f;
            brute = 0.f;
        }
    } confidence;

    struct freestanding_t {
        bool updated{};
        int side{};
        float update_time{};
        float left_fraction{};
        float right_fraction{};

        INLINE void reset() {
            updated = false;
            side = 0;
            update_time = 0.f;
            left_fraction = 0.f;
            right_fraction = 0.f;
        }
    } freestanding;

    anim_record_t record{};

    INLINE void reset() {
        resolved = false;
        side = 0;
        legit_ticks = 0;
        fake_ticks = 0;
        state = resolver_state::IDLE;
        mode.clear();

        jitter.reset();
        ukf->reset();
        particle_filter->init(0.f);
        ttt_net->init();
        bandit.init();

        adaptive_confidence = {};
        skeletal_analyzer = {};
        latency_compensator = {};

        confidence.reset();
        freestanding.reset();

        max_desync = 0.f;
        locked_side = 0;
        lock_time = 0.f;
        brute_step = 0;

        std::fill(brute_offsets.begin(), brute_offsets.end(), 0.f);
        std::fill(brute_hits.begin(), brute_hits.end(), 0);

        for (auto& layer : initial_layers) {
            layer = {};
        }
    }
};

inline std::array<resolver_info_t, 65> resolver_info{};

class resolver_analytics {
public:
    struct tick_data {
        int player_id{};
        float predicted_yaw{};
        float confidence{};
        int chosen_side{};
        bool hit{};
        float timestamp{};
    };

    void log_tick(const tick_data& data);
    void export_csv(const std::string& filename);

private:
    std::vector<tick_data> data_log;
    std::mutex data_mutex;
};

namespace resolver {
    extern resolver_context g_context;
    extern resolver_analytics g_analytics;

    INLINE void reset() {
        for (auto& info : resolver_info) {
            info.reset();
        }
    }

    extern void prepare_side(c_cs_player*, anim_record_t*, anim_record_t*);
    extern void apply_side(c_cs_player*, anim_record_t*, int);
    extern void prepare_side_improved(c_cs_player*, anim_record_t*, anim_record_t*);
    extern void apply_side_improved(c_cs_player*, anim_record_t*, int);
    extern void on_shot_result(c_cs_player*, bool, int, float, float);
    extern void prepare_bones_with_resolver(c_cs_player*, anim_record_t*);

    extern std::array<float, 16> extract_features(c_cs_player*, const resolver_info_t&, anim_record_t*);
    extern void update_context(int tick_rate, float latency, int choked_commands, bool competitive);
}
