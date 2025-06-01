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

template<typename T, size_t N>
struct alignas(64) ring_buffer {
    alignas(64) T data[N]{};
    size_t index{};

    INLINE void push(const T& value) {
        data[index] = value;
        index = (index + 1 == N) ? 0 : index + 1;
    }

    INLINE T& operator[](size_t i) { return data[(index + i) % N]; }
    INLINE const T& operator[](size_t i) const { return data[(index + i) % N]; }
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

struct full_ukf_t {
    static constexpr int N = 5;
    static constexpr int L = 2 * N + 1;

    std::array<float, N> x{};
    std::array<std::array<float, N>, N> P{};
    std::array<float, N> Q{ 1.f, 0.1f, 0.05f, 0.1f, 0.1f };
    std::array<float, 2> R{ 4.f, 4.f };

    std::array<std::array<float, N>, L> sigma_points{};
    std::array<float, L> weights_mean{};
    std::array<float, L> weights_cov{};

    float alpha{ 1e-3f };
    float beta{ 2.f };
    float kappa{ 0.f };

    INLINE void init() {
        float lambda = alpha * alpha * (N + kappa) - N;
        weights_mean[0] = lambda / (N + lambda);
        weights_cov[0] = weights_mean[0] + (1 - alpha * alpha + beta);

        for (int i = 1; i < L; ++i) {
            weights_mean[i] = weights_cov[i] = 0.5f / (N + lambda);
        }
    }

    INLINE void generate_sigma_points() {
        std::copy(x.begin(), x.end(), sigma_points[0].begin());

        std::array<std::array<float, N>, N> sqrt_P{};
        cholesky_decomposition(P, sqrt_P);

        float scale = std::sqrt(N + alpha * alpha * (N + kappa) - N);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                sigma_points[i + 1][j] = x[j] + scale * sqrt_P[j][i];
                sigma_points[i + 1 + N][j] = x[j] - scale * sqrt_P[j][i];
            }
        }
    }

    INLINE void predict() {
        generate_sigma_points();

        for (auto& sp : sigma_points) {
            sp[0] = wrap_deg(sp[0] + sp[1] + 0.5f * sp[2]);
            sp[1] += sp[2];
            sp[0] = wrap_deg(sp[0]);

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

        for (auto& row : P) {
            std::fill(row.begin(), row.end(), 0.f);
        }

        for (int i = 0; i < L; ++i) {
            std::array<float, N> diff{};
            for (int j = 0; j < N; ++j) {
                diff[j] = sigma_points[i][j] - x[j];
            }
            diff[0] = wrap_deg(diff[0]);

            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    P[j][k] += weights_cov[i] * diff[j] * diff[k];
                }
            }
        }

        for (int i = 0; i < N; ++i) {
            P[i][i] += Q[i];
        }
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

        std::array<std::array<float, 2>, 2> S{};
        std::array<std::array<float, 2>, N> C{};

        for (int i = 0; i < L; ++i) {
            std::array<float, 2> dz{ sigma_points[i][3] - z_pred[0], sigma_points[i][4] - z_pred[1] };
            std::array<float, N> dx{};
            for (int j = 0; j < N; ++j) {
                dx[j] = sigma_points[i][j] - x[j];
            }
            dx[0] = wrap_deg(dx[0]);

            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    S[j][k] += weights_cov[i] * dz[j] * dz[k];
                }
            }

            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < 2; ++k) {
                    C[j][k] += weights_cov[i] * dx[j] * dz[k];
                }
            }
        }

        S[0][0] += R[0];
        S[1][1] += R[1];

        std::array<std::array<float, 2>, 2> S_inv{};
        matrix_inverse_2x2(S, S_inv);

        std::array<std::array<float, 2>, N> K{};
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    K[i][j] += C[i][k] * S_inv[k][j];
                }
            }
        }

        std::array<float, 2> y{ z[0] - z_pred[0], z[1] - z_pred[1] };

        for (int i = 0; i < N; ++i) {
            x[i] += K[i][0] * y[0] + K[i][1] * y[1];
        }
        x[0] = wrap_deg(x[0]);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < 2; ++k) {
                    P[i][j] -= K[i][k] * (C[j][0] * K[i][0] + C[j][1] * K[i][1]);
                }
            }
        }
    }

    INLINE float estimate() const { return x[0]; }

    INLINE void reset() {
        std::fill(x.begin(), x.end(), 0.f);
        for (auto& row : P) {
            std::fill(row.begin(), row.end(), 0.f);
        }
        for (int i = 0; i < N; ++i) {
            P[i][i] = 25.f;
        }
        init();
    }

private:
    static INLINE float wrap_deg(float a) {
        a = std::fmod(a + 180.f, 360.f);
        if (a < 0) a += 360.f;
        return a - 180.f;
    }

    static void cholesky_decomposition(const std::array<std::array<float, N>, N>& A, std::array<std::array<float, N>, N>& L) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                float sum = A[i][j];
                for (int k = 0; k < j; ++k) {
                    sum -= L[i][k] * L[j][k];
                }
                if (i == j) {
                    L[i][j] = std::sqrt(std::max(sum, EPSILON));
                }
                else {
                    L[i][j] = sum / L[j][j];
                }
            }
        }
    }

    static void matrix_inverse_2x2(const std::array<std::array<float, 2>, 2>& A, std::array<std::array<float, 2>, 2>& inv) {
        float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
        det = std::max(det, EPSILON);
        float inv_det = 1.f / det;
        inv[0][0] = A[1][1] * inv_det;
        inv[0][1] = -A[0][1] * inv_det;
        inv[1][0] = -A[1][0] * inv_det;
        inv[1][1] = A[0][0] * inv_det;
    }
};

struct particle_filter_t {
    static constexpr int NUM_PARTICLES = 128;

    struct particle {
        float yaw{};
        float weight{ 1.f / NUM_PARTICLES };
    };

    std::array<particle, NUM_PARTICLES> particles{};
    std::mt19937 rng{ std::random_device{}() };
    std::normal_distribution<float> process_noise{ 0.f, 5.f };
    std::normal_distribution<float> measurement_noise{ 0.f, 10.f };

    INLINE void init(float initial_yaw) {
        std::uniform_real_distribution<float> dist(-60.f, 60.f);
        for (auto& p : particles) {
            p.yaw = initial_yaw + dist(rng);
            p.weight = 1.f / NUM_PARTICLES;
        }
    }

    INLINE void predict() {
        for (auto& p : particles) {
            p.yaw += process_noise(rng);
            p.yaw = wrap_deg(p.yaw);
        }
    }

    INLINE void update(float measured_yaw) {
        float weight_sum = 0.f;

        for (auto& p : particles) {
            float diff = wrap_deg(measured_yaw - p.yaw);
            p.weight = std::exp(-0.5f * diff * diff / (measurement_noise.stddev() * measurement_noise.stddev()));
            weight_sum += p.weight;
        }

        if (weight_sum > EPSILON) {
            for (auto& p : particles) {
                p.weight /= weight_sum;
            }
        }

        resample();
    }

    INLINE float estimate() const {
        float weighted_sum = 0.f;
        for (const auto& p : particles) {
            weighted_sum += p.yaw * p.weight;
        }
        return wrap_deg(weighted_sum);
    }

private:
    INLINE void resample() {
        std::array<particle, NUM_PARTICLES> new_particles{};
        std::array<float, NUM_PARTICLES> weights{};

        for (int i = 0; i < NUM_PARTICLES; ++i) {
            weights[i] = particles[i].weight;
        }

        std::discrete_distribution<int> dist(weights.begin(), weights.end());

        for (auto& np : new_particles) {
            np = particles[dist(rng)];
            np.weight = 1.f / NUM_PARTICLES;
        }

        particles = new_particles;
    }

    static INLINE float wrap_deg(float a) {
        a = std::fmod(a + 180.f, 360.f);
        if (a < 0) a += 360.f;
        return a - 180.f;
    }
};

struct neural_resolver_t {
    static constexpr int INPUT_SIZE = 16;
    static constexpr int HIDDEN_SIZE = 32;
    static constexpr int OUTPUT_SIZE = 2;

    std::array<std::array<float, HIDDEN_SIZE>, INPUT_SIZE> w1{};
    std::array<float, HIDDEN_SIZE> b1{};
    std::array<std::array<float, OUTPUT_SIZE>, HIDDEN_SIZE> w2{};
    std::array<float, OUTPUT_SIZE> b2{};

    float learning_rate{ 0.001f };

    INLINE void init() {
        std::mt19937 rng{ std::random_device{}() };
        std::normal_distribution<float> dist(0.f, std::sqrt(2.f / INPUT_SIZE));

        for (auto& row : w1) {
            for (auto& w : row) {
                w = dist(rng);
            }
        }

        dist = std::normal_distribution<float>(0.f, std::sqrt(2.f / HIDDEN_SIZE));
        for (auto& row : w2) {
            for (auto& w : row) {
                w = dist(rng);
            }
        }
    }

    INLINE std::array<float, OUTPUT_SIZE> forward(const std::array<float, INPUT_SIZE>& input) {
        std::array<float, HIDDEN_SIZE> hidden{};

#pragma omp simd
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = b1[i];
            for (int j = 0; j < INPUT_SIZE; ++j) {
                sum += input[j] * w1[j][i];
            }
            hidden[i] = std::max(0.f, sum);
        }

        std::array<float, OUTPUT_SIZE> output{};

#pragma omp simd
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            float sum = b2[i];
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                sum += hidden[j] * w2[j][i];
            }
            output[i] = sum;
        }

        float max_val = *std::max_element(output.begin(), output.end());
        float sum = 0.f;

        for (auto& o : output) {
            o = std::exp(o - max_val);
            sum += o;
        }

        for (auto& o : output) {
            o /= sum;
        }

        return output;
    }

    INLINE void update(const std::array<float, INPUT_SIZE>& input, int true_side, bool hit) {
        float target = hit ? 1.f : 0.f;
        float reward = hit ? 1.f : -0.5f;

        auto output = forward(input);

        std::array<float, OUTPUT_SIZE> grad_output{};
        grad_output[0] = (true_side < 0 ? target : 0.f) - output[0];
        grad_output[1] = (true_side > 0 ? target : 0.f) - output[1];

        for (auto& g : grad_output) {
            g *= reward * learning_rate;
        }

        std::array<float, HIDDEN_SIZE> hidden{};
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = b1[i];
            for (int j = 0; j < INPUT_SIZE; ++j) {
                sum += input[j] * w1[j][i];
            }
            hidden[i] = std::max(0.f, sum);
        }

        std::array<float, HIDDEN_SIZE> grad_hidden{};
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                grad_hidden[i] += grad_output[j] * w2[i][j];
            }
            if (hidden[i] <= 0.f) grad_hidden[i] = 0.f;
        }

        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                w2[i][j] += grad_output[j] * hidden[i];
            }
        }

        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            b2[i] += grad_output[i];
        }

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                w1[i][j] += grad_hidden[j] * input[i];
            }
        }

        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            b1[i] += grad_hidden[i];
        }
    }

    INLINE int predict(const std::array<float, INPUT_SIZE>& input) {
        auto output = forward(input);
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

        float wavelet_re{};
        float wavelet_im{};

        ring_buffer<float, DELTA_WINDOW> delta_history{};
        ring_buffer<float, CACHE_SIZE> delta_cache{};
        ring_buffer<float, YAW_CACHE_SIZE> yaw_cache{};

        float variance{};
        float strength{};
        float autocorr{};
        float delta_variance{};

        int jitter_ticks{};
        int static_ticks{};

        float mean{};
        float m2{};

        alignas(16) float goertzel_q[4]{};
        static constexpr int GOERTZEL_K = 2;

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
            std::memset(goertzel_q, 0, sizeof(goertzel_q));
            mean = 0.f;
            m2 = 0.f;
            wavelet_re = 0.f;
            wavelet_im = 0.f;
        }

        INLINE bool is_high_jitter() const {
            return variance > 15.f && autocorr < 0.f;
        }
    } jitter;

    std::unique_ptr<full_ukf_t> ukf{ std::make_unique<full_ukf_t>() };
    std::unique_ptr<particle_filter_t> particle_filter{ std::make_unique<particle_filter_t>() };
    std::unique_ptr<neural_resolver_t> neural_net{ std::make_unique<neural_resolver_t>() };

    struct contextual_bandit_t {
        static constexpr int NUM_ARMS = 7;
        static constexpr int CONTEXT_DIM = 8;

        struct linear_model_t {
            std::array<float, CONTEXT_DIM> weights{};
            std::array<std::array<float, CONTEXT_DIM>, CONTEXT_DIM> cov{};

            INLINE void init() {
                std::fill(weights.begin(), weights.end(), 0.f);
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    cov[i][i] = 1.f;
                }
            }

            INLINE float predict(const std::array<float, CONTEXT_DIM>& x) const {
                float sum = 0.f;
#pragma omp simd reduction(+:sum)
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    sum += weights[i] * x[i];
                }
                return sum;
            }

            INLINE void update(const std::array<float, CONTEXT_DIM>& x, float reward) {
                std::array<float, CONTEXT_DIM> cov_x{};
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    for (int j = 0; j < CONTEXT_DIM; ++j) {
                        cov_x[i] += cov[i][j] * x[j];
                    }
                }

                float x_cov_x = 0.f;
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    x_cov_x += x[i] * cov_x[i];
                }

                float k_denom = 1.f + x_cov_x;
                std::array<float, CONTEXT_DIM> k{};
                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    k[i] = cov_x[i] / k_denom;
                }

                float pred = predict(x);
                float error = reward - pred;

                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    weights[i] += k[i] * error;
                }

                for (int i = 0; i < CONTEXT_DIM; ++i) {
                    for (int j = 0; j < CONTEXT_DIM; ++j) {
                        cov[i][j] -= k[i] * cov_x[j];
                    }
                }
            }
        };

        std::array<linear_model_t, NUM_ARMS> models{};
        std::mt19937 rng{ std::random_device{}() };
        float exploration_rate{ 0.1f };

        INLINE void init() {
            for (auto& model : models) {
                model.init();
            }
        }

        INLINE int select(const std::array<float, CONTEXT_DIM>& context) {
            std::uniform_real_distribution<float> uniform(0.f, 1.f);

            if (uniform(rng) < exploration_rate) {
                std::uniform_int_distribution<int> arm_dist(0, NUM_ARMS - 1);
                return arm_dist(rng);
            }

            int best_arm = 0;
            float best_value = models[0].predict(context);

            for (int i = 1; i < NUM_ARMS; ++i) {
                float value = models[i].predict(context);
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

            exploration_rate *= 0.995f;
            exploration_rate = std::max(0.01f, exploration_rate);
        }
    } bandit;

    struct adaptive_confidence_t {
        std::array<float, 5> weights{ 0.2f, 0.2f, 0.2f, 0.2f, 0.2f };
        std::array<float, 5> performance{ 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
        float learning_rate{ 0.01f };

        INLINE float blend(const std::array<float, 5>& confidences) {
            float sum = 0.f;
            float weight_sum = 0.f;

            for (int i = 0; i < 5; ++i) {
                float w = weights[i] * performance[i];
                sum += confidences[i] * w;
                weight_sum += w;
            }

            return weight_sum > EPSILON ? sum / weight_sum : 0.5f;
        }

        INLINE void update(int method_idx, bool success) {
            if (method_idx < 0 || method_idx >= 5) return;

            float target = success ? 1.f : 0.f;
            performance[method_idx] += learning_rate * (target - performance[method_idx]);
            performance[method_idx] = std::clamp(performance[method_idx], 0.1f, 0.9f);

            float total_perf = std::accumulate(performance.begin(), performance.end(), 0.f);
            if (total_perf > EPSILON) {
                for (int i = 0; i < 5; ++i) {
                    weights[i] = performance[i] / total_perf;
                }
            }
        }
    } adaptive_confidence;

    struct desync_optimizer_t {
        float current_offset{};
        float best_offset{};
        float search_min{ -60.f };
        float search_max{ 60.f };
        float golden_ratio{ 0.618033988749895f };
        int iterations{};

        INLINE void reset() {
            current_offset = 0.f;
            best_offset = 0.f;
            iterations = 0;
        }

        INLINE float next_test_point() {
            if (iterations < 2) {
                return iterations == 0 ? search_min + golden_ratio * (search_max - search_min)
                    : search_min + (1.f - golden_ratio) * (search_max - search_min);
            }

            float range = search_max - search_min;
            if (range < 1.f) {
                return (search_min + search_max) * 0.5f;
            }

            return current_offset + (range * 0.1f) * (iterations % 2 == 0 ? 1.f : -1.f);
        }

        INLINE void update_result(float offset, bool hit) {
            if (hit) {
                best_offset = offset;
                search_min = offset - 10.f;
                search_max = offset + 10.f;
            }
            else {
                if (offset < current_offset) {
                    search_min = offset;
                }
                else {
                    search_max = offset;
                }
            }

            current_offset = offset;
            iterations++;
        }
    } desync_optimizer;

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
                spine_dir = spine_dir / spine_len;  // Manual normalization
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
        neural_net->init();
        bandit.init();

        adaptive_confidence = {};
        desync_optimizer.reset();
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
