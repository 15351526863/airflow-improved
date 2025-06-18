#include "ctm.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>

// Helper function to normalize angles to [-180, 180]
static float normalize_yaw(float yaw) {
    while (yaw > 180.0f) yaw -= 360.0f;
    while (yaw < -180.0f) yaw += 360.0f;
    return yaw;
}

// Xavier initialization
static void xavier_init(float** weights, int input_dim, int output_dim, std::mt19937& rng) {
    float limit = std::sqrt(6.0f / (input_dim + output_dim));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < input_dim; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            weights[i][j] = dist(rng);
        }
    }
}

// Layer implementation
void ctm_layer_t::init(int in_dim, int out_dim, int hist_size) {
    input_dim = in_dim;
    output_dim = out_dim;
    history_size = hist_size;

    // Allocate weights
    weights = new float* [input_dim];
    weight_momentum = new float* [input_dim];
    for (int i = 0; i < input_dim; ++i) {
        weights[i] = new float[output_dim] {};
        weight_momentum[i] = new float[output_dim] {};
    }

    // Allocate bias
    bias = new float[output_dim] {};
    bias_momentum = new float[output_dim] {};

    // Allocate neurons
    neurons = new ctm_neuron_state_t[output_dim]{};

    // Initialize neuron-level models with history size
    for (int i = 0; i < output_dim; ++i) {
        neurons[i].init_nlm(history_size);
    }

    // Initialize neuron cache
    neuron_cache.resize(output_dim);

    // Initialize input cache
    cached_input.resize(input_dim);

    // Initialize synapse model with proper depth
    synapse.init(input_dim + CTM_INPUT_DIM, 6); // Increased depth for U-Net

    // Initialize weights using Xavier initialization
    std::mt19937 rng(std::random_device{}());
    xavier_init(weights, input_dim, output_dim, rng);
}

void ctm_layer_t::forward(const float* input, float* output, const float* attn_out, bool use_activation) {
    // Cache input for backward pass
    if (input) {
        std::memcpy(cached_input.data(), input, input_dim * sizeof(float));
    }

    // Feed attention output into synapse
    std::vector<float> syn_in(input_dim + CTM_INPUT_DIM);
    std::vector<float> syn_out(input_dim);

    // Copy input
    std::memcpy(syn_in.data(), input, input_dim * sizeof(float));

    // Concatenate attention output if provided
    if (attn_out) {
        std::memcpy(syn_in.data() + input_dim, attn_out, CTM_INPUT_DIM * sizeof(float));
    }
    else {
        // Zero-fill if no attention output
        std::memset(syn_in.data() + input_dim, 0, CTM_INPUT_DIM * sizeof(float));
    }

    // Pass through synapse and save cache
    synapse.forward(syn_in.data(), syn_out.data(), synapse_cache);

    // Compute pre-activations
    for (int j = 0; j < output_dim; ++j) {
        float sum = bias[j];
        // Use synapse output instead of raw input
        for (int i = 0; i < input_dim; ++i) {
            sum += syn_out[i] * weights[i][j];
        }

        if (use_activation) {
            // Push pre-activation into history
            neurons[j].update_history(sum);

            // Run the per-neuron MLP on the last M pre-activations
            output[j] = run_nlm_fwd(neurons[j], neurons[j].history.data(), neuron_cache[j],
                history_size, CTM_NLM_HIDDEN);

            // Update post-activation history for synchronization
            neurons[j].update_post_history(output[j]);
            neurons[j].activation = output[j];
        }
        else {
            output[j] = sum;
        }
    }
}

void ctm_layer_t::backward(const float* input, const float* grad_output, float* grad_input,
    const float* attn_out, float learning_rate) {
    // Compute gradients
    if (grad_input) {
        std::memset(grad_input, 0, input_dim * sizeof(float));
    }

    // Prepare synapse input for backward
    std::vector<float> syn_in(input_dim + CTM_INPUT_DIM);
    std::memcpy(syn_in.data(), input ? input : cached_input.data(), input_dim * sizeof(float));
    if (attn_out) {
        std::memcpy(syn_in.data() + input_dim, attn_out, CTM_INPUT_DIM * sizeof(float));
    }
    else {
        std::memset(syn_in.data() + input_dim, 0, CTM_INPUT_DIM * sizeof(float));
    }

    // Gradient accumulator for synapse output
    std::vector<float> grad_syn_out(input_dim, 0.0f);

    for (int j = 0; j < output_dim; ++j) {
        float grad = grad_output[j];

        // Backpropagate through NLM
        run_nlm_bwd(neurons[j], neurons[j].history.data(), neuron_cache[j],
            grad, learning_rate, history_size, CTM_NLM_HIDDEN);

        // Update bias with momentum
        bias_momentum[j] = CTM_MOMENTUM * bias_momentum[j] + (1.0f - CTM_MOMENTUM) * grad;
        bias[j] -= learning_rate * (bias_momentum[j] + CTM_WEIGHT_DECAY * bias[j]);

        // Update weights with momentum
        for (int i = 0; i < input_dim; ++i) {
            // Use cached input if original input is null
            float input_val = input ? input[i] : cached_input[i];
            float weight_grad = grad * input_val;
            weight_momentum[i][j] = CTM_MOMENTUM * weight_momentum[i][j] + (1.0f - CTM_MOMENTUM) * weight_grad;
            weights[i][j] -= learning_rate * (weight_momentum[i][j] + CTM_WEIGHT_DECAY * weights[i][j]);

            // Accumulate gradient for synapse output
            grad_syn_out[i] += grad * weights[i][j];
        }
    }

    // Backward through synapse
    synapse.backward(syn_in.data(), grad_syn_out.data(), synapse_cache, learning_rate);

    // Copy gradient to grad_input if requested
    if (grad_input) {
        // Note: grad_input should receive gradient w.r.t original input (first part of syn_in)
        // The synapse backward already computed this internally
        std::memcpy(grad_input, grad_syn_out.data(), input_dim * sizeof(float));
    }
}

void ctm_layer_t::update_neurons(float dt) {
    // Basic neuron state update if needed
}

void ctm_layer_t::synchronize_neurons(float coupling_strength) {
    // Synchronization if needed
}

void ctm_layer_t::cleanup() {
    if (weights) {
        for (int i = 0; i < input_dim; ++i) {
            delete[] weights[i];
            delete[] weight_momentum[i];
        }
        delete[] weights;
        delete[] weight_momentum;
    }

    delete[] bias;
    delete[] bias_momentum;
    delete[] neurons;

    // Fix #9: Clear cached vectors
    cached_input.clear();
    neuron_cache.clear();
}

// Thought state implementation
void ctm_thought_state_t::evolve(float dt) {
    think_steps++;
    internal_ticks++;
}

float ctm_thought_state_t::compute_synchrony() const {
    return 0.0f;
}

bool ctm_thought_state_t::is_converged(float threshold) const {
    // Remove synchrony check from convergence criterion
    return think_steps >= CTM_MAX_THINK_STEPS;
}

// CTM implementation
c_continuous_thought_machine::c_continuous_thought_machine(int hist_size) : history_size(hist_size) {
    // Initialize layers with custom history size
    input_layer.init(CTM_INPUT_DIM, CTM_HIDDEN_DIM, history_size);
    hidden_layer.init(CTM_HIDDEN_DIM, CTM_THOUGHT_DIM, history_size);
    thought_layer.init(CTM_THOUGHT_DIM, CTM_HIDDEN_DIM, history_size);
    output_layer.init(CTM_HIDDEN_DIM, CTM_OUTPUT_DIM, history_size);

    // Initialize synchronizers
    sync_out.init(CTM_HIDDEN_DIM, CTM_SYNC_D_OUT);
    sync_action.init(CTM_HIDDEN_DIM, CTM_SYNC_D_ACTION);

    // Initialize gradient accumulators for r
    grad_r_action_accumulated.resize(CTM_SYNC_D_ACTION, 0.0f);
    grad_r_out_accumulated.resize(CTM_SYNC_D_OUT, 0.0f);

    // Initialize attention module
    attn_module.init(CTM_SYNC_D_ACTION, CTM_INPUT_DIM);

    // Initialize learnable initial states
    initial_z1.resize(CTM_HIDDEN_DIM);
    initial_pre_history.resize(CTM_HIDDEN_DIM * history_size);

    // Initialize temporal buffer with custom history size
    temporal_buffer.resize(history_size, std::vector<float>(CTM_INPUT_DIM, 0.0f));

    // Initialize with small random values
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    for (auto& v : initial_z1) v = dist(rng);
    for (auto& v : initial_pre_history) v = dist(rng);

    reset();
}

c_continuous_thought_machine::~c_continuous_thought_machine() {
    input_layer.cleanup();
    hidden_layer.cleanup();
    thought_layer.cleanup();
    output_layer.cleanup();
}

void c_continuous_thought_machine::forward(const float* input, float* output) {
    // Use std::vector instead of static buffers
    std::vector<float> hidden1(CTM_HIDDEN_DIM);
    std::vector<float> thought(CTM_THOUGHT_DIM);
    std::vector<float> hidden2(CTM_HIDDEN_DIM);
    std::vector<float> attn_out(CTM_INPUT_DIM, 0.0f); // Attention output placeholder

    // Forward through layers with attention output
    input_layer.forward(input, hidden1.data(), attn_out.data());
    hidden_layer.forward(hidden1.data(), thought.data(), attn_out.data());

    // Update thought state
    for (int i = 0; i < CTM_THOUGHT_DIM; ++i) {
        thought_state.state[i] = thought[i];
    }

    thought_layer.forward(thought_state.state, hidden2.data(), attn_out.data());
    output_layer.forward(hidden2.data(), output, attn_out.data(), false); // No activation on output

    // Store logits for certainty-aware loss
    if (training_mode && thought_state.internal_ticks < MAX_INTERNAL_TICKS) {
        std::memcpy(logits_history[thought_state.internal_ticks], output,
            sizeof(float) * CTM_OUTPUT_DIM);

        // Compute and store entropy
        float max_logit = *std::max_element(output, output + CTM_OUTPUT_DIM);
        float sum_exp = 0.0f;
        for (int i = 0; i < CTM_OUTPUT_DIM; ++i) {
            sum_exp += std::exp(output[i] - max_logit);
        }
        float log_sum = std::log(sum_exp) + max_logit;

        float entropy = 0.0f;
        for (int i = 0; i < CTM_OUTPUT_DIM; ++i) {
            float p = std::exp(output[i] - log_sum);
            if (p > 1e-8f) {
                entropy -= p * std::log(p);
            }
        }
        // Use modulo for entropy history index
        entropy_history[thought_state.internal_ticks % MAX_INTERNAL_TICKS] = entropy / std::log(float(CTM_OUTPUT_DIM));
    }
}

void c_continuous_thought_machine::think(int max_steps) {
    thought_state.reset();

    // Clear gradient accumulators
    std::fill(grad_r_action_accumulated.begin(), grad_r_action_accumulated.end(), 0.0f);
    std::fill(grad_r_out_accumulated.begin(), grad_r_out_accumulated.end(), 0.0f);

    while (!thought_state.is_converged() && thought_state.think_steps < max_steps) {
        // Early exit based on entropy/certainty only
        if (thought_state.internal_ticks > 0 && thought_state.internal_ticks < MAX_INTERNAL_TICKS) {
            // Use modulo for entropy history access
            float ent = entropy_history[(thought_state.internal_ticks - 1) % MAX_INTERNAL_TICKS];
            if (ent < 0.15f) {  // High certainty, stop thinking
                break;
            }
        }

        // Evolve thought state
        thought_state.evolve();

        // Build z_vec from all neurons' activations
        std::vector<float> z_vec(CTM_HIDDEN_DIM);
        for (int i = 0; i < CTM_HIDDEN_DIM; ++i) {
            z_vec[i] = thought_layer.neurons[i].activation;
        }

        // Compute action synchronization
        std::vector<float> sync_a(CTM_SYNC_D_ACTION);
        std::vector<float> grad_r(CTM_SYNC_D_ACTION); // For r gradients
        sync_action.compute(thought_layer.neurons, sync_a.data(), grad_r.data());

        // Store gradients for later update (after loss is computed)
        for (size_t i = 0; i < grad_r.size(); ++i) {
            grad_r_action_accumulated[i] += grad_r[i];
        }

        // Also compute and accumulate output synchronization gradients
        std::vector<float> sync_o(CTM_SYNC_D_OUT);
        std::vector<float> grad_r_o(CTM_SYNC_D_OUT);
        sync_out.compute(thought_layer.neurons, sync_o.data(), grad_r_o.data());

        for (size_t i = 0; i < grad_r_o.size(); ++i) {
            grad_r_out_accumulated[i] += grad_r_o[i];
        }

        // Use sync_a as query to attention
        std::vector<float> attn_out(CTM_HIDDEN_DIM, 0.0f);
        // In full implementation, this would interact with external data

        // Process through synapse model with attention output
        std::vector<float> synapse_input(CTM_HIDDEN_DIM * 2);
        std::memcpy(synapse_input.data(), z_vec.data(), CTM_HIDDEN_DIM * sizeof(float));
        std::memcpy(synapse_input.data() + CTM_HIDDEN_DIM, attn_out.data(),
            CTM_HIDDEN_DIM * sizeof(float));

        std::vector<float> synapse_out(CTM_HIDDEN_DIM);
        thought_layer.synapse.forward(synapse_input.data(), synapse_out.data(), thought_layer.synapse_cache);

        // Feed synapse output back through NLMs to obtain z_{t+1}
        for (int j = 0; j < CTM_HIDDEN_DIM; ++j) {
            thought_layer.neurons[j].update_history(synapse_out[j]);
            float z_next = run_nlm_fwd(thought_layer.neurons[j],
                thought_layer.neurons[j].history.data(),
                thought_layer.neuron_cache[j],
                history_size, CTM_NLM_HIDDEN);
            thought_layer.neurons[j].update_post_history(z_next);
            thought_layer.neurons[j].activation = z_next;
        }

        // Update neural dynamics
        input_layer.update_neurons();
        hidden_layer.update_neurons();
        thought_layer.update_neurons();
        output_layer.update_neurons();

        // Synchronize neurons
        hidden_layer.synchronize_neurons();
        thought_layer.synchronize_neurons();
    }
}

void c_continuous_thought_machine::backward(const float* target, float* output) {
    if (!training_mode) return;

    int current_tick = thought_state.internal_ticks;
    if (current_tick >= MAX_INTERNAL_TICKS) return;

    // For classification task - assume target is class index
    int target_class = static_cast<int>(target[0]);

    // Lambda for computing softmax log sum
    auto softmax_logsum = [](const float* logits) {
        float mx = *std::max_element(logits, logits + CTM_OUTPUT_DIM);
        float sum = 0.0f;
        for (int i = 0; i < CTM_OUTPUT_DIM; ++i) {
            sum += std::exp(logits[i] - mx);
        }
        return std::log(sum) + mx;
        };

    // If this is the last internal tick, compute certainty-aware loss
    if (current_tick == thought_state.think_steps - 1) {
        int t1 = 0, t2 = 0;
        float min_loss = 1e9f, max_certainty = -1.0f;

        for (int t = 0; t < current_tick + 1; ++t) {
            // Compute per-tick log_sum
            float t_log_sum = softmax_logsum(logits_history[t]);
            float nll = -logits_history[t][target_class] + t_log_sum;

            // Use modulo for entropy history access
            float certainty = 1.0f - entropy_history[t % MAX_INTERNAL_TICKS];

            if (nll < min_loss) {
                min_loss = nll;
                t1 = t;
            }
            if (certainty > max_certainty) {
                max_certainty = certainty;
                t2 = t;
            }
        }

        // Compute loss from t1 and t2
        float log_sum_t1 = softmax_logsum(logits_history[t1]);
        float log_sum_t2 = softmax_logsum(logits_history[t2]);
        float loss_t1 = -logits_history[t1][target_class] + log_sum_t1;
        float loss_t2 = -logits_history[t2][target_class] + log_sum_t2;
        float final_loss = 0.5f * (loss_t1 + loss_t2);

        total_loss += final_loss;
        loss_count++;

        // Compute gradients
        std::vector<float> grad_output(CTM_OUTPUT_DIM);
        for (int i = 0; i < CTM_OUTPUT_DIM; ++i) {
            float p1 = std::exp(logits_history[t1][i] - log_sum_t1);
            float p2 = std::exp(logits_history[t2][i] - log_sum_t2);
            grad_output[i] = 0.5f * (p1 + p2);
            if (i == target_class) {
                grad_output[i] -= 1.0f;
            }
            grad_output[i] /= CTM_OUTPUT_DIM;
        }

        // Backward through layers
        std::vector<float> grad_hidden2(CTM_HIDDEN_DIM);
        std::vector<float> grad_thought(CTM_THOUGHT_DIM);
        std::vector<float> grad_hidden1(CTM_HIDDEN_DIM);
        std::vector<float> attn_out(CTM_INPUT_DIM, 0.0f);

        output_layer.backward(nullptr, grad_output.data(), grad_hidden2.data(),
            attn_out.data(), CTM_LEARNING_RATE);
        thought_layer.backward(thought_state.state, grad_hidden2.data(), grad_thought.data(),
            attn_out.data(), CTM_LEARNING_RATE);
        hidden_layer.backward(nullptr, grad_thought.data(), grad_hidden1.data(),
            attn_out.data(), CTM_LEARNING_RATE);
        input_layer.backward(nullptr, grad_hidden1.data(), nullptr,
            attn_out.data(), CTM_LEARNING_RATE);

        // Fix #3: Apply chain rule for r gradients and update
        // We need gradient of loss w.r.t S (synchronization output)
        // For simplicity, assume uniform gradient distribution
        std::vector<float> grad_loss_sync_action(CTM_SYNC_D_ACTION, 0.1f);
        std::vector<float> grad_loss_sync_out(CTM_SYNC_D_OUT, 0.1f);

        // Multiply accumulated gradients by loss gradient
        for (size_t i = 0; i < grad_r_action_accumulated.size(); ++i) {
            grad_r_action_accumulated[i] *= grad_loss_sync_action[i];
        }
        for (size_t i = 0; i < grad_r_out_accumulated.size(); ++i) {
            grad_r_out_accumulated[i] *= grad_loss_sync_out[i];
        }

        // Update r values with proper gradients
        sync_action.update_r(grad_r_action_accumulated.data(), CTM_LEARNING_RATE);
        sync_out.update_r(grad_r_out_accumulated.data(), CTM_LEARNING_RATE);
    }
}

void c_continuous_thought_machine::update_weights(float learning_rate) {
    // Weights are updated during backward pass with momentum
}

void c_continuous_thought_machine::add_temporal_input(const float* input) {
    std::memcpy(temporal_buffer[temporal_idx % history_size].data(), input, CTM_INPUT_DIM * sizeof(float));
    temporal_idx++;
}

void c_continuous_thought_machine::process_temporal_sequence() {
    // Process temporal buffer to extract temporal features
    std::vector<float> temporal_features(CTM_INPUT_DIM, 0.0f);

    // Compute temporal average and variance
    for (int j = 0; j < CTM_INPUT_DIM; ++j) {
        float sum = 0.0f, sum_sq = 0.0f;
        for (int i = 0; i < history_size; ++i) {
            sum += temporal_buffer[i][j];
            sum_sq += temporal_buffer[i][j] * temporal_buffer[i][j];
        }
        float mean = sum / history_size;
        float var = (sum_sq / history_size) - (mean * mean);
        temporal_features[j] = std::sqrt(std::max(0.0f, var));
    }

    // Use temporal features in forward pass
    std::vector<float> output(CTM_OUTPUT_DIM);
    forward(temporal_features.data(), output.data());
}

void c_continuous_thought_machine::reset() {
    input_layer.reset_neurons();
    hidden_layer.reset_neurons();
    thought_layer.reset_neurons();
    output_layer.reset_neurons();

    // Initialize neurons with learnable initial states
    for (int i = 0; i < CTM_HIDDEN_DIM; ++i) {
        thought_layer.neurons[i].activation = initial_z1[i];
        // Initialize pre-activation history
        for (int j = 0; j < history_size; ++j) {
            thought_layer.neurons[i].history[j] = initial_pre_history[i * history_size + j];
        }
    }

    thought_state.reset();
    temporal_idx = 0;

    // Clear temporal buffer
    for (auto& buf : temporal_buffer) {
        std::fill(buf.begin(), buf.end(), 0.0f);
    }

    std::memset(logits_history, 0, sizeof(logits_history));
    std::memset(entropy_history, 0, sizeof(entropy_history));

    // Clear gradient accumulators
    std::fill(grad_r_action_accumulated.begin(), grad_r_action_accumulated.end(), 0.0f);
    std::fill(grad_r_out_accumulated.begin(), grad_r_out_accumulated.end(), 0.0f);

    total_loss = 0.0f;
    loss_count = 0;
}

void c_continuous_thought_machine::save_weights(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return;

    // Save history size
    file.write(reinterpret_cast<const char*>(&history_size), sizeof(int));

    // Save layer weights
    auto save_layer = [&](const ctm_layer_t& layer) {
        file.write(reinterpret_cast<const char*>(&layer.input_dim), sizeof(int));
        file.write(reinterpret_cast<const char*>(&layer.output_dim), sizeof(int));

        for (int i = 0; i < layer.input_dim; ++i) {
            file.write(reinterpret_cast<const char*>(layer.weights[i]), layer.output_dim * sizeof(float));
            // Save momentum buffers
            file.write(reinterpret_cast<const char*>(layer.weight_momentum[i]), layer.output_dim * sizeof(float));
        }
        file.write(reinterpret_cast<const char*>(layer.bias), layer.output_dim * sizeof(float));
        // Save bias momentum
        file.write(reinterpret_cast<const char*>(layer.bias_momentum), layer.output_dim * sizeof(float));

        // Save NLM parameters
        for (int i = 0; i < layer.output_dim; ++i) {
            const auto& neuron = layer.neurons[i];
            file.write(reinterpret_cast<const char*>(neuron.w1.data()),
                neuron.w1.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(neuron.b1.data()),
                neuron.b1.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(neuron.w2.data()),
                neuron.w2.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(&neuron.b2), sizeof(float));

            // Save NLM momentum
            file.write(reinterpret_cast<const char*>(neuron.w1_momentum.data()),
                neuron.w1_momentum.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(neuron.b1_momentum.data()),
                neuron.b1_momentum.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(neuron.w2_momentum.data()),
                neuron.w2_momentum.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(&neuron.b2_momentum), sizeof(float));
        }

        // Save synapse weights and momentum
        for (int idx = 0; idx < layer.synapse.k; ++idx) {
            file.write(reinterpret_cast<const char*>(layer.synapse.w[idx].data()),
                layer.synapse.w[idx].size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(layer.synapse.b[idx].data()),
                layer.synapse.b[idx].size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(layer.synapse.w_mom[idx].data()),
                layer.synapse.w_mom[idx].size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(layer.synapse.b_mom[idx].data()),
                layer.synapse.b_mom[idx].size() * sizeof(float));
        }
        };

    save_layer(input_layer);
    save_layer(hidden_layer);
    save_layer(thought_layer);
    save_layer(output_layer);

    // Save synchronizer decay parameters
    file.write(reinterpret_cast<const char*>(&sync_out.pairs[0]),
        sync_out.pairs.size() * sizeof(sync_pair_t));
    file.write(reinterpret_cast<const char*>(&sync_action.pairs[0]),
        sync_action.pairs.size() * sizeof(sync_pair_t));

    // Save learnable initial states
    file.write(reinterpret_cast<const char*>(initial_z1.data()),
        initial_z1.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(initial_pre_history.data()),
        initial_pre_history.size() * sizeof(float));

    file.close();
}

void c_continuous_thought_machine::load_weights(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return;

    // Load history size
    int saved_history_size;
    file.read(reinterpret_cast<char*>(&saved_history_size), sizeof(int));

    // Verify history size matches
    if (saved_history_size != history_size) {
        file.close();
        return;
    }

    // Load layer weights
    auto load_layer = [&](ctm_layer_t& layer) {
        int in_dim, out_dim;
        file.read(reinterpret_cast<char*>(&in_dim), sizeof(int));
        file.read(reinterpret_cast<char*>(&out_dim), sizeof(int));

        if (in_dim != layer.input_dim || out_dim != layer.output_dim) return;

        for (int i = 0; i < layer.input_dim; ++i) {
            file.read(reinterpret_cast<char*>(layer.weights[i]), layer.output_dim * sizeof(float));
            // Load momentum buffers
            file.read(reinterpret_cast<char*>(layer.weight_momentum[i]), layer.output_dim * sizeof(float));
        }
        file.read(reinterpret_cast<char*>(layer.bias), layer.output_dim * sizeof(float));
        // Load bias momentum
        file.read(reinterpret_cast<char*>(layer.bias_momentum), layer.output_dim * sizeof(float));

        // Load NLM parameters
        for (int i = 0; i < layer.output_dim; ++i) {
            auto& neuron = layer.neurons[i];
            file.read(reinterpret_cast<char*>(neuron.w1.data()),
                neuron.w1.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(neuron.b1.data()),
                neuron.b1.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(neuron.w2.data()),
                neuron.w2.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(&neuron.b2), sizeof(float));

            // Load NLM momentum
            file.read(reinterpret_cast<char*>(neuron.w1_momentum.data()),
                neuron.w1_momentum.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(neuron.b1_momentum.data()),
                neuron.b1_momentum.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(neuron.w2_momentum.data()),
                neuron.w2_momentum.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(&neuron.b2_momentum), sizeof(float));
        }

        // Load synapse weights and momentum
        for (int layer_idx = 0; layer_idx < layer.synapse.k; ++layer_idx) {
            file.read(reinterpret_cast<char*>(layer.synapse.w[layer_idx].data()),
                layer.synapse.w[layer_idx].size() * sizeof(float));
            file.read(reinterpret_cast<char*>(layer.synapse.b[layer_idx].data()),
                layer.synapse.b[layer_idx].size() * sizeof(float));
            file.read(reinterpret_cast<char*>(layer.synapse.w_mom[layer_idx].data()),
                layer.synapse.w_mom[layer_idx].size() * sizeof(float));
            file.read(reinterpret_cast<char*>(layer.synapse.b_mom[layer_idx].data()),
                layer.synapse.b_mom[layer_idx].size() * sizeof(float));
        }
        };

    load_layer(input_layer);
    load_layer(hidden_layer);
    load_layer(thought_layer);
    load_layer(output_layer);

    // Load synchronizer decay parameters
    file.read(reinterpret_cast<char*>(&sync_out.pairs[0]),
        sync_out.pairs.size() * sizeof(sync_pair_t));
    file.read(reinterpret_cast<char*>(&sync_action.pairs[0]),
        sync_action.pairs.size() * sizeof(sync_pair_t));

    // Load learnable initial states
    file.read(reinterpret_cast<char*>(initial_z1.data()),
        initial_z1.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(initial_pre_history.data()),
        initial_pre_history.size() * sizeof(float));

    file.close();
}