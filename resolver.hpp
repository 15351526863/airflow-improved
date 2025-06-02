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

    INLINE T& operator[](size_t i) { return data[(index + N - 1 - i) % N]; }
    INLINE const T& operator[](size_t i) const { return data[(index + N - 1 - i) % N]; }

    INLINE T& get_chronological(size_t i) {
        return data[(index + N - 1 - i) % N];
    }
    INLINE const T& get_chronological(size_t i) const {
        return data[(index + N - 1 - i) % N];
    }
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

    float alpha{ 0.1f };
    float beta{ 2.f };
    float kappa{ 3.f - float(N) };
    float dt{ 1.f / 64.f };

    INLINE void init() {
        float lambda = alpha * alpha * (N + kappa) - N;
        weights_mean[0] = lambda / (N + lambda);
        weights_cov[0] = weights_mean[0] + (1 - alpha * alpha + beta);

        for (int i = 1; i < L; ++i) {
            weights_mean[i] = weights_cov[i] = 0.5f / (N + lambda);
        }
    }

    INLINE void set_dt(float delta_time) {
        dt = delta_time;
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
                float sum = 0.f;
                for (int k = 0; k < 2; ++k) {
                    for (int l = 0; l < 2; ++l) {
                        sum += K[i][k] * S[k][l] * K[j][l];
                    }
                }
                P[i][j] -= sum;
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
        if (std::fabs(det) < EPSILON) {
            det = (det >= 0.f ? 1.f : -1.f) * EPSILON;
        }
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
        rng.seed(std::random_device{}());
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
        float s = 0.f, c = 0.f;
        for (const auto& p : particles) {
            float rad = DEG2RAD(p.yaw);
            s += std::sin(rad) * p.weight;
            c += std::cos(rad) * p.weight;
        }
        return RAD2DEG(std::atan2(s, c));
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

struct kan_resolver_t {
    static constexpr int INPUT_SIZE = 16;
    static constexpr int HIDDEN_SIZE = 32;
    static constexpr int OUTPUT_SIZE = 2;
    static constexpr int SPLINE_ORDER = 4;
    static constexpr int NUM_KNOTS = 8;
    static constexpr int NUM_COEFFS = NUM_KNOTS + SPLINE_ORDER - 1;

    struct spline_edge {
        std::array<float, NUM_COEFFS> coeffs{};
        float grid_min{ -2.f };
        float grid_max{ 2.f };

        INLINE float evaluate(float x) const {
            x = std::clamp(x, grid_min, grid_max);
            float t = (x - grid_min) / (grid_max - grid_min) * (NUM_KNOTS - 1);
            int k = std::min(int(t), NUM_KNOTS - 2);
            t -= k;

            float b0 = (1.f - t) * (1.f - t) * (1.f - t) / 6.f;
            float b1 = (3.f * t * t * t - 6.f * t * t + 4.f) / 6.f;
            float b2 = (-3.f * t * t * t + 3.f * t * t + 3.f * t + 1.f) / 6.f;
            float b3 = t * t * t / 6.f;

            return coeffs[k] * b0 + coeffs[k + 1] * b1 +
                coeffs[k + 2] * b2 + coeffs[k + 3] * b3;
        }

        INLINE std::pair<std::array<float, 4>, std::array<float, 4>> evaluate_with_basis_and_deriv(float x, int& k_out) const {
            x = std::clamp(x, grid_min, grid_max);
            float grid_range = grid_max - grid_min;
            float t = (x - grid_min) / grid_range * (NUM_KNOTS - 1);
            int k = std::min(int(t), NUM_KNOTS - 2);
            k_out = k;
            t -= k;

            std::array<float, 4> basis{};
            basis[0] = (1.f - t) * (1.f - t) * (1.f - t) / 6.f;
            basis[1] = (3.f * t * t * t - 6.f * t * t + 4.f) / 6.f;
            basis[2] = (-3.f * t * t * t + 3.f * t * t + 3.f * t + 1.f) / 6.f;
            basis[3] = t * t * t / 6.f;

            float dt_dx = (NUM_KNOTS - 1) / grid_range;

            std::array<float, 4> basis_deriv{};
            basis_deriv[0] = -0.5f * (1.f - t) * (1.f - t) * dt_dx;
            basis_deriv[1] = (1.5f * t * t - 2.f * t) * dt_dx;
            basis_deriv[2] = (-1.5f * t * t + t + 0.5f) * dt_dx;
            basis_deriv[3] = 0.5f * t * t * dt_dx;

            return { basis, basis_deriv };
        }
    };

    std::array<std::array<spline_edge, HIDDEN_SIZE>, INPUT_SIZE> layer1_edges{};
    std::array<std::array<spline_edge, OUTPUT_SIZE>, HIDDEN_SIZE> layer2_edges{};

    float learning_rate{ 0.001f };
    float l1_lambda{ 0.01f };

    std::mt19937 rng{ std::random_device{}() };

    INLINE void init() {
        std::normal_distribution<float> dist(0.f, 0.1f);

        for (auto& input_edges : layer1_edges) {
            for (auto& edge : input_edges) {
                for (auto& coeff : edge.coeffs) {
                    coeff = dist(rng);
                }
                edge.grid_min = -2.f;
                edge.grid_max = 2.f;
            }
        }

        for (auto& hidden_edges : layer2_edges) {
            for (auto& edge : hidden_edges) {
                for (auto& coeff : edge.coeffs) {
                    coeff = dist(rng);
                }
                edge.grid_min = -2.f;
                edge.grid_max = 2.f;
            }
        }
    }

    INLINE std::array<float, OUTPUT_SIZE> forward(const std::array<float, INPUT_SIZE>& input) {
        std::array<float, HIDDEN_SIZE> hidden{};

#pragma omp simd
        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            float sum = 0.f;
            for (int i = 0; i < INPUT_SIZE; ++i) {
                sum += layer1_edges[i][h].evaluate(input[i]);
            }
            hidden[h] = sum;
        }

        std::array<float, OUTPUT_SIZE> output{};

#pragma omp simd
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            float sum = 0.f;
            for (int h = 0; h < HIDDEN_SIZE; ++h) {
                sum += layer2_edges[h][o].evaluate(hidden[h]);
            }
            output[o] = sum;
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

    INLINE void update(const std::array<float, INPUT_SIZE>& input, int true_side, bool hit) {
        auto output = forward(input);

        std::array<float, OUTPUT_SIZE> grad_output{};
        if (true_side != 0) {
            float target0 = true_side < 0 ? 1.f : 0.f;
            float target1 = true_side > 0 ? 1.f : 0.f;

            grad_output[0] = output[0] - target0;
            grad_output[1] = output[1] - target1;
        }
        else {
            float entropy_grad_scale = 0.1f;
            grad_output[0] = entropy_grad_scale * (output[0] - 0.5f);
            grad_output[1] = entropy_grad_scale * (output[1] - 0.5f);
        }

        std::array<float, HIDDEN_SIZE> hidden{};
        std::array<std::array<int, HIDDEN_SIZE>, INPUT_SIZE> layer1_k{};
        std::array<std::array<std::pair<std::array<float, 4>, std::array<float, 4>>, HIDDEN_SIZE>, INPUT_SIZE> layer1_basis_deriv{};

        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            float sum = 0.f;
            for (int i = 0; i < INPUT_SIZE; ++i) {
                int k;
                auto [basis, deriv] = layer1_edges[i][h].evaluate_with_basis_and_deriv(input[i], k);
                layer1_k[i][h] = k;
                layer1_basis_deriv[i][h] = { basis, deriv };

                for (int b = 0; b < 4; ++b) {
                    sum += layer1_edges[i][h].coeffs[k + b] * basis[b];
                }
            }
            hidden[h] = sum;
        }

        std::array<std::array<int, OUTPUT_SIZE>, HIDDEN_SIZE> layer2_k{};
        std::array<std::array<std::pair<std::array<float, 4>, std::array<float, 4>>, OUTPUT_SIZE>, HIDDEN_SIZE> layer2_basis_deriv{};

        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            for (int h = 0; h < HIDDEN_SIZE; ++h) {
                int k;
                auto [basis, deriv] = layer2_edges[h][o].evaluate_with_basis_and_deriv(hidden[h], k);
                layer2_k[h][o] = k;
                layer2_basis_deriv[h][o] = { basis, deriv };
            }
        }

        std::array<float, HIDDEN_SIZE> grad_hidden{};

        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            for (int o = 0; o < OUTPUT_SIZE; ++o) {
                int k = layer2_k[h][o];
                auto& [basis, deriv] = layer2_basis_deriv[h][o];

                for (int b = 0; b < 4; ++b) {
                    float grad = grad_output[o] * basis[b];
                    layer2_edges[h][o].coeffs[k + b] -= learning_rate * grad;

                    float coeff_value = layer2_edges[h][o].coeffs[k + b];
                    if (std::abs(coeff_value) > EPSILON) {
                        float l1_grad = l1_lambda * (coeff_value > 0 ? 1.f : -1.f);
                        float new_value = coeff_value - learning_rate * l1_grad;

                        if (coeff_value * new_value < 0.f) {
                            layer2_edges[h][o].coeffs[k + b] = 0.f;
                        }
                        else {
                            layer2_edges[h][o].coeffs[k + b] = new_value;
                        }
                    }
                }

                float spline_deriv_sum = 0.f;
                for (int b = 0; b < 4; ++b) {
                    spline_deriv_sum += layer2_edges[h][o].coeffs[k + b] * deriv[b];
                }
                grad_hidden[h] += grad_output[o] * spline_deriv_sum;
            }
        }

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int h = 0; h < HIDDEN_SIZE; ++h) {
                int k = layer1_k[i][h];
                auto& [basis, deriv] = layer1_basis_deriv[i][h];

                for (int b = 0; b < 4; ++b) {
                    float grad = grad_hidden[h] * basis[b];
                    layer1_edges[i][h].coeffs[k + b] -= learning_rate * grad;

                    float coeff_value = layer1_edges[i][h].coeffs[k + b];
                    if (std::abs(coeff_value) > EPSILON) {
                        float l1_grad = l1_lambda * (coeff_value > 0 ? 1.f : -1.f);
                        float new_value = coeff_value - learning_rate * l1_grad;

                        if (coeff_value * new_value < 0.f) {
                            layer1_edges[i][h].coeffs[k + b] = 0.f;
                        }
                        else {
                            layer1_edges[i][h].coeffs[k + b] = new_value;
                        }
                    }
                }
            }
        }

        float total_l1 = 0.f;
        for (const auto& input_edges : layer1_edges) {
            for (const auto& edge : input_edges) {
                for (float coeff : edge.coeffs) {
                    total_l1 += std::abs(coeff);
                }
            }
        }

        if (total_l1 > 100.f) {
            float scale = 100.f / total_l1;
            for (auto& input_edges : layer1_edges) {
                for (auto& edge : input_edges) {
                    for (auto& coeff : edge.coeffs) {
                        coeff *= scale;
                    }
                }
            }
        }
    }

    INLINE int predict(const std::array<float, INPUT_SIZE>& input) {
        auto output = forward(input);
        return output[1] > output[0] ? 1 : -1;
    }

    INLINE void adapt_grid_ranges(const std::array<float, INPUT_SIZE>& input) {
        float adapt_rate = 0.01f;

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int h = 0; h < HIDDEN_SIZE; ++h) {
                auto& edge = layer1_edges[i][h];
                edge.grid_min += adapt_rate * (input[i] - edge.grid_min);
                edge.grid_max += adapt_rate * (input[i] - edge.grid_max);

                edge.grid_min = std::min(edge.grid_min, -0.1f);
                edge.grid_max = std::max(edge.grid_max, 0.1f);
            }
        }
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

        alignas(16) float goertzel_q0{};
        alignas(16) float goertzel_q1{};
        alignas(16) float goertzel_q2{};
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
            goertzel_q0 = 0.f;
            goertzel_q1 = 0.f;
            goertzel_q2 = 0.f;
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
    std::unique_ptr<kan_resolver_t> kan_net{ std::make_unique<kan_resolver_t>() };

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
                        cov[i][j] -= k[i] * x[j];
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
                float w = weights[i];
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
        kan_net->init();
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
