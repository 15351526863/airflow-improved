#pragma once
#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>
#include <vector>
#include <random>

#define INLINE __forceinline

constexpr int CTM_INPUT_DIM = 64;      // Input features dimension
constexpr int CTM_HIDDEN_DIM = 128;    // Hidden layer dimension
constexpr int CTM_THOUGHT_DIM = 32;    // Thought/synchronization dimension
constexpr int CTM_OUTPUT_DIM = 8;      // Output dimension
constexpr int CTM_DEFAULT_HISTORY_SIZE = 32;   // Default temporal history size (M in paper)
constexpr int CTM_MAX_THINK_STEPS = 8; // Maximum thinking steps
constexpr float CTM_LEARNING_RATE = 0.001f;
constexpr float CTM_MOMENTUM = 0.9f;
constexpr float CTM_WEIGHT_DECAY = 0.0001f;
constexpr int CTM_NLM_HIDDEN = 64;     // Hidden size for neuron-level models (H in paper)
constexpr int CTM_SYNC_D_OUT = 8196;   // D_out for output synchronization
constexpr int CTM_SYNC_D_ACTION = 2048;// D_action for action synchronization

// Activation functions
INLINE float ctm_relu(float x) { return std::max(0.0f, x); }
INLINE float ctm_leaky_relu(float x, float alpha = 0.01f) { return x > 0 ? x : alpha * x; }
INLINE float ctm_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-std::clamp(x, -10.0f, 10.0f))); }
INLINE float ctm_tanh(float x) { return std::tanh(x); }

// Derivatives
INLINE float ctm_relu_derivative(float x) { return x > 0 ? 1.0f : 0.0f; }
INLINE float ctm_leaky_relu_derivative(float x, float alpha = 0.01f) { return x > 0 ? 1.0f : alpha; }
INLINE float ctm_sigmoid_derivative(float y) { return y * (1.0f - y); }
INLINE float ctm_tanh_derivative(float y) { return 1.0f - y * y; }

// Cache structure for NLM forward pass
struct nlm_cache_t {
    std::vector<float> hidden;
    nlm_cache_t() : hidden(CTM_NLM_HIDDEN) {}
};

// Cache structure for synapse forward pass
struct synapse_cache_t {
    std::vector<std::vector<float>> activations;
};

struct ctm_neuron_state_t {
    float activation{};

    // Pre-activation history (for NLM input)
    std::vector<float> history;
    int history_idx{};
    int history_size{};

    // Neuron-Level Model (NLM) parameters
    std::vector<float> w1;      // shape [M * H]
    std::vector<float> b1;      // shape [H]
    std::vector<float> w2;      // shape [H]
    float b2{};

    // NLM momentum terms for training
    std::vector<float> w1_momentum;
    std::vector<float> b1_momentum;
    std::vector<float> w2_momentum;
    float b2_momentum{};

    // Post-activation history (for synchronization)
    std::vector<float> post_history;  // Dynamic size based on current tick

    void init_nlm(int M = CTM_DEFAULT_HISTORY_SIZE, int H = CTM_NLM_HIDDEN) {
        history_size = M;
        history.resize(M, 0.0f);

        w1.resize(M * H);
        b1.resize(H);
        w2.resize(H);

        // Initialize momentum terms
        w1_momentum.resize(M * H, 0.0f);
        b1_momentum.resize(H, 0.0f);
        w2_momentum.resize(H, 0.0f);
        b2_momentum = 0.0f;

        // Xavier initialization
        std::mt19937 rng{ std::random_device{}() };
        float limit1 = std::sqrt(6.0f / (M + H));
        float limit2 = std::sqrt(6.0f / (H + 1));
        std::uniform_real_distribution<float> dist1(-limit1, limit1);
        std::uniform_real_distribution<float> dist2(-limit2, limit2);

        for (int i = 0; i < M * H; ++i) w1[i] = dist1(rng);
        for (int i = 0; i < H; ++i) {
            b1[i] = dist1(rng);
            w2[i] = dist2(rng);
        }
        b2 = dist2(rng);
    }

    INLINE void update_history(float pre_activation) {
        history[history_idx % history_size] = pre_activation;
        history_idx++;
    }

    INLINE void update_post_history(float post_activation) {
        post_history.push_back(post_activation);
    }

    INLINE void reset() {
        activation = 0.0f;
        history_idx = 0;
        std::fill(history.begin(), history.end(), 0.0f);
        post_history.clear();
    }
};

// Forward pass for NLM with caching
INLINE float run_nlm_fwd(const ctm_neuron_state_t& n,
    const float* history,
    nlm_cache_t& cache,
    int M, int H) {
    // (1) hidden = ReLU( history · W1 + b1 )
    for (int h = 0; h < H; ++h) {
        cache.hidden[h] = n.b1[h];
        for (int t = 0; t < M; ++t) {
            cache.hidden[h] += history[t] * n.w1[t * H + h];
        }
        cache.hidden[h] = ctm_relu(cache.hidden[h]);
    }

    // (2) out = hidden · W2 + b2
    float out = n.b2;
    for (int h = 0; h < H; ++h) {
        out += cache.hidden[h] * n.w2[h];
    }
    return out;
}

// Backward pass for NLM
INLINE void run_nlm_bwd(ctm_neuron_state_t& n,
    const float* history,
    const nlm_cache_t& cache,
    float d_out,
    float lr,
    int M, int H) {
    // Gradient for b2
    n.b2_momentum = CTM_MOMENTUM * n.b2_momentum + (1.0f - CTM_MOMENTUM) * d_out;
    n.b2 -= lr * (n.b2_momentum + CTM_WEIGHT_DECAY * n.b2);

    // Gradient for W2 and backprop to hidden
    std::vector<float> d_hidden(H, 0.0f);
    for (int h = 0; h < H; ++h) {
        float grad = d_out * cache.hidden[h];
        n.w2_momentum[h] = CTM_MOMENTUM * n.w2_momentum[h] + (1.0f - CTM_MOMENTUM) * grad;
        n.w2[h] -= lr * (n.w2_momentum[h] + CTM_WEIGHT_DECAY * n.w2[h]);

        // Backprop to hidden
        d_hidden[h] = d_out * n.w2[h];
        // ReLU derivative
        if (cache.hidden[h] <= 0) d_hidden[h] = 0;
    }

    // Gradient for b1
    for (int h = 0; h < H; ++h) {
        n.b1_momentum[h] = CTM_MOMENTUM * n.b1_momentum[h] + (1.0f - CTM_MOMENTUM) * d_hidden[h];
        n.b1[h] -= lr * (n.b1_momentum[h] + CTM_WEIGHT_DECAY * n.b1[h]);
    }

    // Gradient for W1
    for (int t = 0; t < M; ++t) {
        for (int h = 0; h < H; ++h) {
            float grad = d_hidden[h] * history[t];
            int idx = t * H + h;
            n.w1_momentum[idx] = CTM_MOMENTUM * n.w1_momentum[idx] + (1.0f - CTM_MOMENTUM) * grad;
            n.w1[idx] -= lr * (n.w1_momentum[idx] + CTM_WEIGHT_DECAY * n.w1[idx]);
        }
    }
}

// Synapse model (U-Net style with depth parameter)
struct ctm_synapse_t {
    int in_dim{}, out_dim{}, k{}; // k = depth
    std::vector<std::vector<float>> w;
    std::vector<std::vector<float>> b;
    std::vector<int> layer_dims;

    // Momentum buffers for training
    std::vector<std::vector<float>> w_mom;
    std::vector<std::vector<float>> b_mom;

    void init(int d, int depth = 4) {
        in_dim = out_dim = d;
        k = depth;

        // Setup layer dimensions (contract then expand) - Fixed according to error #5
        layer_dims.resize(k + 1);
        layer_dims[0] = d;

        // Calculate bottleneck dimension
        int bottleneck = 16;
        int step = (d - bottleneck) / (k / 2);

        // Contracting path
        for (int i = 1; i <= k / 2; ++i) {
            layer_dims[i] = std::max(bottleneck, layer_dims[i - 1] - step);
        }

        // Expanding path (mirror)
        for (int i = k / 2 + 1; i <= k; ++i) {
            layer_dims[i] = layer_dims[k - i];
        }

        // Allocate weights, biases and momentum
        w.resize(k);
        b.resize(k);
        w_mom.resize(k);
        b_mom.resize(k);

        std::mt19937 rng{ std::random_device{}() };
        for (int layer = 0; layer < k; ++layer) {
            int curr_dim = layer_dims[layer];
            int next_dim = layer_dims[layer + 1];

            w[layer].resize(curr_dim * next_dim);
            b[layer].resize(next_dim);
            w_mom[layer].assign(curr_dim * next_dim, 0.0f);
            b_mom[layer].assign(next_dim, 0.0f);

            // Xavier init
            float limit = std::sqrt(6.0f / (curr_dim + next_dim));
            std::uniform_real_distribution<float> dist(-limit, limit);

            for (auto& v : w[layer]) v = dist(rng);
            for (auto& v : b[layer]) v = dist(rng);
        }
    }

    void forward(const float* in, float* out, synapse_cache_t& cache) const {
        // Allocate buffers for intermediate activations
        cache.activations.resize(k + 1);
        cache.activations[0].assign(in, in + in_dim);

        // Forward through contracting path
        for (int layer = 0; layer < k / 2; ++layer) {
            int curr_dim = layer_dims[layer];
            int next_dim = layer_dims[layer + 1];
            cache.activations[layer + 1].resize(next_dim);

            for (int j = 0; j < next_dim; ++j) {
                float sum = b[layer][j];
                for (int i = 0; i < curr_dim; ++i) {
                    sum += cache.activations[layer][i] * w[layer][i * next_dim + j];
                }
                cache.activations[layer + 1][j] = ctm_relu(sum);
            }
        }

        // Forward through expanding path with skip connections
        for (int layer = k / 2; layer < k; ++layer) {
            int curr_dim = layer_dims[layer];
            int next_dim = layer_dims[layer + 1];
            int skip_layer = k - layer - 1;
            cache.activations[layer + 1].resize(next_dim);

            for (int j = 0; j < next_dim; ++j) {
                float sum = b[layer][j];
                for (int i = 0; i < curr_dim; ++i) {
                    sum += cache.activations[layer][i] * w[layer][i * next_dim + j];
                }
                // Add skip connection
                if (skip_layer >= 0 && skip_layer < layer && j < cache.activations[skip_layer].size()) {
                    sum += cache.activations[skip_layer][j];
                }
                cache.activations[layer + 1][j] = (layer == k - 1) ? sum : ctm_relu(sum);
            }
        }

        // Copy final output
        std::memcpy(out, cache.activations[k].data(), out_dim * sizeof(float));
    }

    void backward(const float* in, const float* grad_out, const synapse_cache_t& cache, float lr) {
        std::vector<std::vector<float>> grad_acts(k + 1);
        grad_acts[k].assign(grad_out, grad_out + out_dim);

        // Backward through expanding path
        for (int layer = k - 1; layer >= k / 2; --layer) {
            int curr_dim = layer_dims[layer];
            int next_dim = layer_dims[layer + 1];
            int skip_layer = k - layer - 1;
            grad_acts[layer].resize(curr_dim, 0.0f);

            for (int j = 0; j < next_dim; ++j) {
                float grad = grad_acts[layer + 1][j];

                // Apply ReLU derivative (except for last layer)
                if (layer != k - 1 && cache.activations[layer + 1][j] <= 0) {
                    grad = 0;
                }

                // Update bias
                b_mom[layer][j] = CTM_MOMENTUM * b_mom[layer][j] + (1.0f - CTM_MOMENTUM) * grad;
                b[layer][j] -= lr * (b_mom[layer][j] + CTM_WEIGHT_DECAY * b[layer][j]);

                // Update weights and propagate gradient
                for (int i = 0; i < curr_dim; ++i) {
                    size_t idx = i * next_dim + j;
                    w_mom[layer][idx] = CTM_MOMENTUM * w_mom[layer][idx] +
                        (1.0f - CTM_MOMENTUM) * grad * cache.activations[layer][i];
                    w[layer][idx] -= lr * (w_mom[layer][idx] + CTM_WEIGHT_DECAY * w[layer][idx]);
                    grad_acts[layer][i] += w[layer][idx] * grad;
                }

                // Propagate gradient through skip connection
                if (skip_layer >= 0 && skip_layer < layer && j < grad_acts[skip_layer].size()) {
                    if (grad_acts[skip_layer].empty()) grad_acts[skip_layer].resize(layer_dims[skip_layer], 0.0f);
                    grad_acts[skip_layer][j] += grad;
                }
            }
        }

        // Backward through contracting path
        for (int layer = k / 2 - 1; layer >= 0; --layer) {
            int curr_dim = layer_dims[layer];
            int next_dim = layer_dims[layer + 1];
            grad_acts[layer].resize(curr_dim, 0.0f);

            for (int j = 0; j < next_dim; ++j) {
                float grad = grad_acts[layer + 1][j];

                // Apply ReLU derivative
                if (cache.activations[layer + 1][j] <= 0) {
                    grad = 0;
                }

                // Update bias
                b_mom[layer][j] = CTM_MOMENTUM * b_mom[layer][j] + (1.0f - CTM_MOMENTUM) * grad;
                b[layer][j] -= lr * (b_mom[layer][j] + CTM_WEIGHT_DECAY * b[layer][j]);

                // Update weights and propagate gradient
                for (int i = 0; i < curr_dim; ++i) {
                    size_t idx = i * next_dim + j;
                    w_mom[layer][idx] = CTM_MOMENTUM * w_mom[layer][idx] +
                        (1.0f - CTM_MOMENTUM) * grad * cache.activations[layer][i];
                    w[layer][idx] -= lr * (w_mom[layer][idx] + CTM_WEIGHT_DECAY * w[layer][idx]);
                    grad_acts[layer][i] += w[layer][idx] * grad;
                }
            }
        }
    }
};

// Synchronization pair with momentum for r
struct sync_pair_t {
    int i, j;
    float r;           // Learnable exponential decay
    float r_momentum;  // Momentum for r updates
};

// Neural synchronizer
struct ctm_synchronizer_t {
    std::vector<sync_pair_t> pairs;        // Sampled (i,j) pairs

    void init(int D, int D_sel) {
        std::mt19937 rng{ std::random_device{}() };
        std::uniform_int_distribution<int> uni(0, D - 1);
        pairs.resize(D_sel);
        for (auto& p : pairs) {
            p.i = uni(rng);
            // Fix #4: Exclude self-pairs
            do {
                p.j = uni(rng);
            } while (p.j == p.i);
            p.r = 0.0f;           // Initialize decay to 0
            p.r_momentum = 0.0f;  // Initialize momentum to 0
        }
    }

    // Compute ∑ exp(-r Δt) z_i[t] z_j[t] / √(∑exp(-r Δt))
    float compute_pair(const ctm_neuron_state_t& ni,
        const ctm_neuron_state_t& nj,
        float r,
        float* d_r = nullptr) const {
        if (ni.post_history.empty() || nj.post_history.empty())
            return 0.0f;

        int len = std::min(ni.post_history.size(), nj.post_history.size());
        float num = 0.0f, den = 0.0f, decay = 1.0f;
        float sum_deltas = 0.0f; // For gradient computation

        // Process from most recent to oldest
        for (int k = len - 1; k >= 0; --k) {
            float delta_t = float(len - 1 - k);
            decay = std::exp(-r * delta_t);

            float prod = ni.post_history[k] * nj.post_history[k];
            num += decay * prod;
            den += decay;

            if (d_r) {
                sum_deltas += decay * delta_t * prod;
            }
        }

        float result = num / std::sqrt(den + 1e-6f);

        if (d_r) {
            // Compute gradient ∂S/∂r
            *d_r = -sum_deltas / std::sqrt(den + 1e-6f);
        }

        return result;
    }

    void compute(const ctm_neuron_state_t* neurons,
        float* out,
        float* grad_r = nullptr) const {
        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            const auto& p = pairs[idx];
            float d_r = 0.0f;
            out[idx] = compute_pair(neurons[p.i], neurons[p.j], p.r,
                grad_r ? &d_r : nullptr);
            if (grad_r) {
                grad_r[idx] = d_r;
            }
        }
    }

    // Update r values with gradients (including chain rule)
    void update_r(const float* grad_loss_r, float lr) {
        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            auto& p = pairs[idx];
            p.r_momentum = CTM_MOMENTUM * p.r_momentum + (1.0f - CTM_MOMENTUM) * grad_loss_r[idx];
            p.r -= lr * p.r_momentum;
            p.r = std::max(0.0f, p.r); // Keep r non-negative
        }
    }
};

// Simple attention module for CTM
struct ctm_attention_t {
    int query_dim{}, key_dim{}, value_dim{}, num_heads{};
    std::vector<float> w_q, w_k, w_v, w_o;

    // Momentum buffers for training
    std::vector<float> w_q_mom, w_k_mom, w_v_mom, w_o_mom;

    void init(int q_dim, int kv_dim, int heads = 8) {
        query_dim = q_dim;
        key_dim = value_dim = kv_dim;
        num_heads = heads;

        int head_dim = query_dim / num_heads;
        w_q.resize(query_dim * query_dim);
        w_k.resize(key_dim * query_dim);
        w_v.resize(value_dim * query_dim);
        w_o.resize(query_dim * query_dim);

        // Initialize momentum buffers
        w_q_mom.resize(query_dim * query_dim, 0.0f);
        w_k_mom.resize(key_dim * query_dim, 0.0f);
        w_v_mom.resize(value_dim * query_dim, 0.0f);
        w_o_mom.resize(query_dim * query_dim, 0.0f);

        // Xavier init
        std::mt19937 rng{ std::random_device{}() };
        float scale = std::sqrt(6.0f / (query_dim + key_dim));
        std::uniform_real_distribution<float> dist(-scale, scale);

        for (auto* w : { &w_q, &w_k, &w_v, &w_o })
            for (auto& v : *w) v = dist(rng);
    }

    void forward(const float* query, const float* keys, const float* values,
        float* output, int kv_len) {
        // Simplified multi-head attention
        int head_dim = query_dim / num_heads;

        // Project Q, K, V
        std::vector<float> q(query_dim), k(kv_len * query_dim), v(kv_len * query_dim);

        // Q = query @ W_q
        for (int i = 0; i < query_dim; ++i) {
            q[i] = 0;
            for (int j = 0; j < query_dim; ++j)
                q[i] += query[j] * w_q[j * query_dim + i];
        }

        // K = keys @ W_k, V = values @ W_v
        for (int t = 0; t < kv_len; ++t) {
            for (int i = 0; i < query_dim; ++i) {
                k[t * query_dim + i] = 0;
                v[t * query_dim + i] = 0;
                for (int j = 0; j < key_dim; ++j) {
                    k[t * query_dim + i] += keys[t * key_dim + j] * w_k[j * query_dim + i];
                    v[t * query_dim + i] += values[t * value_dim + j] * w_v[j * query_dim + i];
                }
            }
        }

        // Compute attention and output
        std::vector<float> attn_out(query_dim, 0);
        float scale = 1.0f / std::sqrt(float(head_dim));

        for (int h = 0; h < num_heads; ++h) {
            int h_start = h * head_dim;
            int h_end = (h + 1) * head_dim;

            // Compute attention scores
            std::vector<float> scores(kv_len);
            for (int t = 0; t < kv_len; ++t) {
                scores[t] = 0;
                for (int i = h_start; i < h_end; ++i)
                    scores[t] += q[i] * k[t * query_dim + i];
                scores[t] *= scale;
            }

            // Softmax - Fix #8: add epsilon to denominator
            float max_score = *std::max_element(scores.begin(), scores.end());
            float sum_exp = 0;
            for (auto& s : scores) {
                s = std::exp(s - max_score);
                sum_exp += s;
            }
            sum_exp += 1e-6f;  // Fix division by zero
            for (auto& s : scores) s /= sum_exp;

            // Weighted sum of values
            for (int i = h_start; i < h_end; ++i) {
                for (int t = 0; t < kv_len; ++t)
                    attn_out[i] += scores[t] * v[t * query_dim + i];
            }
        }

        // Output projection
        for (int i = 0; i < query_dim; ++i) {
            output[i] = 0;
            for (int j = 0; j < query_dim; ++j)
                output[i] += attn_out[j] * w_o[j * query_dim + i];
        }
    }
};

struct ctm_layer_t {
    int input_dim{};
    int output_dim{};
    int history_size{};
    float** weights{};           // [input_dim][output_dim]
    float* bias{};               // [output_dim]
    float** weight_momentum{};   // [input_dim][output_dim]
    float* bias_momentum{};      // [output_dim]
    ctm_neuron_state_t* neurons{}; // [output_dim]

    // Cache for backward pass
    std::vector<float> cached_input;

    // Cache for NLM forward pass
    std::vector<nlm_cache_t> neuron_cache;

    // Cache for synapse forward pass
    synapse_cache_t synapse_cache;

    // Synapse model for this layer
    ctm_synapse_t synapse;

    void init(int in_dim, int out_dim, int hist_size = CTM_DEFAULT_HISTORY_SIZE);
    void forward(const float* input, float* output, const float* attn_out = nullptr, bool use_activation = true);
    void backward(const float* input, const float* grad_output, float* grad_input, const float* attn_out, float learning_rate);
    void update_neurons(float dt = 0.1f);
    void synchronize_neurons(float coupling_strength = 0.1f);
    void cleanup();

    INLINE void reset_neurons() {
        for (int i = 0; i < output_dim; ++i)
            neurons[i].reset();
    }
};

struct ctm_thought_state_t {
    float state[CTM_THOUGHT_DIM]{};
    int think_steps{};
    int internal_ticks{};  // Total internal ticks

    INLINE void reset() {
        std::memset(state, 0, sizeof(state));
        think_steps = 0;
        internal_ticks = 0;
    }

    void evolve(float dt = 0.1f);
    float compute_synchrony() const;
    bool is_converged(float threshold = 0.95f) const;
};

class c_continuous_thought_machine {
private:
    // Network layers
    ctm_layer_t input_layer{};
    ctm_layer_t hidden_layer{};
    ctm_layer_t thought_layer{};
    ctm_layer_t output_layer{};

    // Synchronizers
    ctm_synchronizer_t sync_out{};    // For output projection
    ctm_synchronizer_t sync_action{}; // For attention query

    // Attention module
    ctm_attention_t attn_module{};

    // Temporal processing
    ctm_thought_state_t thought_state{};
    int history_size{};
    std::vector<std::vector<float>> temporal_buffer;
    int temporal_idx{};

    // Training state
    bool training_mode{ false };
    float total_loss{};
    int loss_count{};

    // For certainty-aware loss
    static constexpr int MAX_INTERNAL_TICKS = 50;
    float logits_history[MAX_INTERNAL_TICKS][CTM_OUTPUT_DIM]{};
    float entropy_history[MAX_INTERNAL_TICKS]{};

    // For gradient computation of r values
    std::vector<float> grad_r_action_accumulated;
    std::vector<float> grad_r_out_accumulated;

    // Learnable initial states
    std::vector<float> initial_z1;  // Initial post-activations
    std::vector<float> initial_pre_history; // Initial pre-activation history

    // Random number generator
    std::mt19937 rng{ std::random_device{}() };

public:
    c_continuous_thought_machine(int history_size = CTM_DEFAULT_HISTORY_SIZE);
    ~c_continuous_thought_machine();

    // Core functionality
    void forward(const float* input, float* output);
    void think(int max_steps = CTM_MAX_THINK_STEPS);
    void backward(const float* target, float* output);
    void update_weights(float learning_rate = CTM_LEARNING_RATE);

    // Temporal processing
    void add_temporal_input(const float* input);
    void process_temporal_sequence();

    // Training control
    void set_training(bool mode) { training_mode = mode; }
    float get_loss() const { return loss_count > 0 ? total_loss / loss_count : 0.0f; }
    void reset_loss() { total_loss = 0.0f; loss_count = 0; }

    // State management
    void reset();
    void save_weights(const char* filename);
    void load_weights(const char* filename);
};