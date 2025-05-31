#include "globals.hpp"
#include "resolver.hpp"
#include "animations.hpp"
#include "server_bones.hpp"
#include "ragebot.hpp"
#include "defines.hpp"
#include <algorithm>
#include <cmath>

namespace resolver
{
    inline float logistic(float x);
    inline float circular_mean(const float* angles, int count)
    {
        float sum_sin = 0.f, sum_cos = 0.f;
        for (int i = 0; i < count; ++i)
        {
            float a = DEG2RAD(angles[i]);
            sum_sin += std::sinf(a);
            sum_cos += std::cosf(a);
        }
        return RAD2DEG(std::atan2f(sum_sin / count, sum_cos / count));
    }

    inline float circular_variance(const float* angles, int count, float mean_deg)
    {
        float sum_sin = 0.f, sum_cos = 0.f;
        for (int i = 0; i < count; ++i)
        {
            const float a = DEG2RAD(angles[i]);
            sum_sin += std::sinf(a);
            sum_cos += std::cosf(a);
        }
        const float R = std::sqrt(sum_sin * sum_sin + sum_cos * sum_cos) / count;
        return 1.f - R;
    }

    inline void analyze_jitter_pattern(resolver_info_t::jitter_info_t& jitter)
    {
        float avg = 0.f, deriv = 0.f;
        for (int i = 0; i < resolver_info_t::jitter_info_t::DELTA_WINDOW; ++i)
        {
            const int next = (i + 1) % resolver_info_t::jitter_info_t::DELTA_WINDOW;
            const float cur = jitter.delta_history[i];
            const float nxt = jitter.delta_history[next];
            avg += cur;
            deriv += math::normalize_yaw(nxt - cur);
        }
        avg /= static_cast<float>(resolver_info_t::jitter_info_t::DELTA_WINDOW);
        deriv /= static_cast<float>(resolver_info_t::jitter_info_t::DELTA_WINDOW);

        float best_corr = 0.f;
        int best_lag = 1;
        for (int lag = 1; lag <= 4; ++lag)
        {
            float corr = 0.f;
            for (int i = 0; i < resolver_info_t::jitter_info_t::DELTA_WINDOW; ++i)
            {
                const int next = (i + lag) % resolver_info_t::jitter_info_t::DELTA_WINDOW;
                corr += jitter.delta_history[i] * jitter.delta_history[next];
            }
            corr = std::fabs(corr);
            if (corr > best_corr)
            {
                best_corr = corr;
                best_lag = lag;
            }
        }
        best_corr /= static_cast<float>(resolver_info_t::jitter_info_t::DELTA_WINDOW);

        float mean_angle = circular_mean(jitter.delta_history,
            resolver_info_t::jitter_info_t::DELTA_WINDOW);
        jitter.variance = circular_variance(jitter.delta_history,
            resolver_info_t::jitter_info_t::DELTA_WINDOW, mean_angle) * 100.f;
        jitter.autocorr = best_corr;
        jitter.delta_variance = std::fabs(deriv);

        jitter.high_freq_jitter = best_lag <= 2 && best_corr > 5.f;
        if (jitter.variance > 10.f && jitter.delta_variance < 5.f)
            jitter.is_jitter = true;
    }

    constexpr float FFT_TWO_PI = 2.f * static_cast<float>(M_PI);

    inline int detect_jitter_frequency(const resolver_info_t::jitter_info_t& j)
    {
        constexpr int N = resolver_info_t::jitter_info_t::DELTA_WINDOW;
        float best_power = 0.f;
        int best_band = 0;

        for (int k = 1; k <= N / 4; ++k)
        {
            float band_real = 0.f, band_imag = 0.f;
            for (int n = 0; n < N; ++n)
            {
                float a = FFT_TWO_PI * static_cast<float>(k * n) / N;
                float val = j.delta_history[n];
                band_real += val * std::cosf(a);
                band_imag += val * std::sinf(a);
            }
            float power = band_real * band_real + band_imag * band_imag;
            if (power > best_power)
            {
                best_power = power;
                best_band = k;
            }
        }

        if (best_band == 0)
            return 0; // DC
        if (best_band == 1)
            return 1; // fundamental
        return 2;
    }

    inline void analyze_jitter_spectrum(resolver_info_t::jitter_info_t& jitter)
    {
        const int N = resolver_info_t::jitter_info_t::DELTA_WINDOW;

        float best_amp = 0.f;
        int best_k = -1;

        for (int k = 1; k <= N / 2; ++k)
        {
            float real = 0.f;
            float imag = 0.f;
            for (int n = 0; n < N; ++n)
            {
                float angle = FFT_TWO_PI * static_cast<float>(k * n) / static_cast<float>(N);
                float value = jitter.delta_history[n];
                real += value * std::cosf(angle);
                imag += value * std::sinf(angle);
            }
            float amp = std::sqrtf(real * real + imag * imag);
            if (amp > best_amp)
            {
                best_amp = amp;
                best_k = k;
            }
        }

        best_amp /= static_cast<float>(N);
        jitter.autocorr = best_amp;
        jitter.high_freq_jitter = best_k >= (N / 4);
        jitter.is_jitter = jitter.is_jitter || best_amp > 5.f;
    }

    inline void analyze_jitter_spectrum_weighted(resolver_info_t::jitter_info_t& j)
    {
        constexpr int N = resolver_info_t::jitter_info_t::DELTA_WINDOW;
        thread_local float window[N];

        for (int i = 0; i < N; ++i)
        {
            float t = static_cast<float>(i) / static_cast<float>(N - 1);
            window[i] = 0.5f - 0.5f * std::cosf(FFT_TWO_PI * t);
        }

        float best_power = 0.f;
        int best_band = 0;

        for (int k = 1; k <= N / 2; ++k)
        {
            float real = 0.f, imag = 0.f;
            for (int n = 0; n < N; ++n)
            {
                float angle = FFT_TWO_PI * static_cast<float>(k * n) / static_cast<float>(N);
                float v = j.delta_history[n] * window[n];
                real += v * std::cosf(angle);
                imag += v * std::sinf(angle);
            }
            float power = real * real + imag * imag;
            if (power > best_power)
            {
                best_power = power;
                best_band = k;
            }
        }

        best_power /= static_cast<float>(N) * static_cast<float>(N);
        j.autocorr = best_power;
        j.high_freq_jitter = best_band >= (N / 4);
        j.is_jitter = j.is_jitter || best_power > 10.f;
    }

    inline void analyze_jitter_fft(resolver_info_t::jitter_info_t& j)
    {
        constexpr int N = resolver_info_t::jitter_info_t::DELTA_WINDOW;
        thread_local float window[N];

        for (int i = 0; i < N; ++i)
        {
            float t = static_cast<float>(i) / static_cast<float>(N - 1);
            window[i] = 0.54f - 0.46f * std::cosf(FFT_TWO_PI * t);
        }

        float best_power = 0.f;
        int best_band = 0;

        for (int k = 1; k <= N / 2; ++k)
        {
            float real = 0.f, imag = 0.f;
            for (int n = 0; n < N; ++n)
            {
                float angle = FFT_TWO_PI * static_cast<float>(k * n) /
                    static_cast<float>(N);
                float value = j.delta_history[n] * window[n];
                real += value * std::cosf(angle);
                imag += value * std::sinf(angle);
            }

            float power = real * real + imag * imag;
            if (power > best_power)
            {
                best_power = power;
                best_band = k;
            }
        }

        best_power /= static_cast<float>(N) * static_cast<float>(N);
        j.autocorr = best_power;
        j.high_freq_jitter = best_band >= (N / 4);
        j.is_jitter = j.is_jitter || best_power > 8.f;
        j.frequency = best_band;
    }

    inline void analyze_jitter_derivative(resolver_info_t::jitter_info_t& j)
    {
        constexpr int N = resolver_info_t::jitter_info_t::DELTA_WINDOW;
        int pos = 0, neg = 0;

        for (int i = 0; i < N - 1; ++i)
        {
            int next = (i + 1) % N;
            float d = math::normalize_yaw(j.delta_history[next] -
                j.delta_history[i]);

            if (d > 0.f) ++pos;
            else if (d < 0.f) ++neg;
        }

        float balance = std::fabsf(static_cast<float>(pos - neg));
        if (balance < N * 0.2f && (pos + neg) > N / 2)
            j.is_jitter = true;
    }

    inline void analyze_jitter_frequency(resolver_info_t::jitter_info_t& j)
    {
        constexpr int N = resolver_info_t::jitter_info_t::DELTA_WINDOW;

        float newest = j.delta_history[(j.delta_offset + N - 1) % N];
        float oldest = j.delta_history[j.delta_offset % N];
        float total = math::normalize_yaw(newest - oldest);

        float dt = static_cast<float>(N) * HACKS->global_vars->interval_per_tick;
        float turns = std::fabs(total) / 360.f;
        float freq = turns / std::max(dt, 0.001f);

        j.frequency = freq > 2.f ? 2 : (freq > 0.5f ? 1 : 0);
    }

    inline void update_jitter_fft(resolver_info_t::jitter_info_t& j, float newest)
    {
        constexpr int N = resolver_info_t::jitter_info_t::DELTA_WINDOW;
        const float old = j.delta_history[j.delta_offset % N];
        for (int k = 1; k <= N / 4; ++k)
        {
            const float angle = FFT_TWO_PI * static_cast<float>(k) / static_cast<float>(N);
            const float cos_a = std::cosf(angle);
            const float sin_a = std::sinf(angle);

            const float r = j.fft_real[k];
            const float im = j.fft_imag[k];

            const float new_r = r * cos_a - im * sin_a + newest - old;
            const float new_im = r * sin_a + im * cos_a;

            j.fft_real[k] = new_r;
            j.fft_imag[k] = new_im;
        }

        float best_power = 0.f;
        int best_band = 0;
        for (int k = 1; k <= N / 4; ++k)
        {
            float p = j.fft_real[k] * j.fft_real[k] + j.fft_imag[k] * j.fft_imag[k];
            if (p > best_power)
            {
                best_power = p;
                best_band = k;
            }
        }

        j.frequency = best_band > 1 ? (best_band >= 3 ? 2 : 1) : 0;
        j.high_freq_jitter |= best_power / static_cast<float>(N * N) > 6.f;
    }

    inline void update_jitter_state(resolver_info_t::jitter_info_t& j)
    {
        float freq_w = j.frequency == 2 ? 1.f : (j.frequency == 1 ? 0.5f : 0.f);
        float activity = j.variance + j.delta_variance + freq_w * 5.f;

        if (activity > 15.f)
        {
            j.jitter_ticks = std::min(j.jitter_ticks + 1, 3);
            j.static_ticks = std::max(0, j.static_ticks - 1);
        }
        else
        {
            j.static_ticks = std::min(j.static_ticks + 1, 3);
            j.jitter_ticks = std::max(0, j.jitter_ticks - 1);
        }

        j.is_jitter = j.jitter_ticks > j.static_ticks;
        j.machine.update(j.is_jitter, j.variance);
        j.is_jitter = j.machine.is_jitter();
    }

    inline void analyze_jitter_crosscorr(resolver_info_t::jitter_info_t& jitter)
    {
        constexpr int N = resolver_info_t::jitter_info_t::DELTA_WINDOW;
        constexpr int HALF_N = N / 2;

        float corr_positive = 0.f;
        float corr_negative = 0.f;

        for (int i = 0; i < HALF_N; ++i)
        {
            int idx1 = (jitter.delta_offset + i) % N;
            int idx2 = (jitter.delta_offset + i + HALF_N) % N;

            float d1 = jitter.delta_history[idx1];
            float d2 = jitter.delta_history[idx2];

            corr_positive += std::fabs(d1 - d2);
            corr_negative += std::fabs(d1 + d2);
        }

        float score = corr_positive - corr_negative;
        jitter.is_jitter |= score > 10.f;
        jitter.high_freq_jitter |= score > 20.f;
    }

    static INLINE float weighted_mean(const float* arr, const float* w, int n)
    {
        float s = 0.f, ws = 0.f;
        for (int i = 0; i < n; ++i)
        {
            s += arr[i] * w[i];
            ws += w[i];
        }
        return ws > 0.f ? s / ws : 0.f;
    }

    inline float exponential_mean(const float* arr, int N, int write_off, float alpha)
    {
        float mean = 0.f, weight = 1.f, sum_w = 0.f;
        for (int i = 0; i < N; ++i)
        {
            const int idx = (write_off + N - 1 - i) % N;
            mean += arr[idx] * weight;
            sum_w += weight;
            weight *= alpha;
        }
        return mean / std::max(sum_w, 0.001f);
    }

    inline void analyze_jitter_exponential(resolver_info_t::jitter_info_t& j)
    {
        constexpr int N = resolver_info_t::jitter_info_t::DELTA_WINDOW;

        float mean = exponential_mean(j.delta_history, N, j.delta_offset, 0.85f);
        float var = 0.f;
        float weight = 1.f;
        float sum_w = 0.f;
        for (int i = 0; i < N; ++i)
        {
            int idx = (j.delta_offset + N - 1 - i) % N;
            float d = j.delta_history[idx] - mean;
            var += d * d * weight;
            sum_w += weight;
            weight *= 0.85f;
        }
        var /= std::max(sum_w, 0.001f);

        j.variance = var;
        j.is_jitter |= var > 12.f;
    }

    inline void analyze_jitter_unified(resolver_info_t::jitter_info_t& j)
    {
        constexpr int N = resolver_info_t::jitter_info_t::DELTA_WINDOW;
        thread_local float win[N];
        for (int i = 0; i < N; ++i)
        {
            float t = static_cast<float>(i) / static_cast<float>(N - 1);
            win[i] = 0.5f - 0.5f * std::cosf(FFT_TWO_PI * t);
        }

        float mean = weighted_mean(j.delta_history, win, N);
        float var = 0.f;
        for (int i = 0; i < N; ++i)
        {
            float d = j.delta_history[i] - mean;
            var += win[i] * d * d;
        }
        var /= N;

        float best_pow = 0.f;
        int best_k = 0;
        for (int k = 1; k <= N / 2; ++k)
        {
            float real = 0.f, imag = 0.f;
            for (int n = 0; n < N; ++n)
            {
                float ang = FFT_TWO_PI * static_cast<float>(k * n) / N;
                float v = j.delta_history[n] * win[n];
                real += v * std::cosf(ang);
                imag += v * std::sinf(ang);
            }
            float p = real * real + imag * imag;
            if (p > best_pow)
            {
                best_pow = p;
                best_k = k;
            }
        }
        best_pow /= static_cast<float>(N) * static_cast<float>(N);

        int pos = 0, neg = 0;
        for (int i = 0; i < N - 1; ++i)
        {
            float d = math::normalize_yaw(j.delta_history[(i + 1) % N] - j.delta_history[i]);
            if (d > 0.f) ++pos;
            else if (d < 0.f) ++neg;
        }
        float balance = std::fabsf(static_cast<float>(pos - neg));

        float freq_w = logistic(best_pow * 0.5f) * 2.f - 1.f;
        float bal_w = logistic(5.f - balance) * 2.f - 1.f;
        float total = var * 0.5f + best_pow * freq_w + bal_w;

        j.autocorr = best_pow;
        j.variance = var;
        j.delta_variance = balance;
        j.high_freq_jitter = best_k >= (N / 4);
        j.is_jitter = total > 10.f;
    }
    inline void prepare_jitter(c_cs_player* player, resolver_info_t& info, anim_record_t* current, anim_record_t* last)
    {
        auto& jitter = info.jitter;

        if (last) {
            float delta = math::normalize_yaw(current->eye_angles.y - last->eye_angles.y);
            info.desync_estimator.update(delta, player->velocity().length_2d());

            jitter.delta_cache[jitter.cache_offset % CACHE_SIZE] = delta;
            jitter.cache_offset = (jitter.cache_offset + 1) % CACHE_SIZE;

            jitter.delta_history[jitter.delta_offset % resolver_info_t::jitter_info_t::DELTA_WINDOW] = delta;
            jitter.delta_offset = (jitter.delta_offset + 1) % resolver_info_t::jitter_info_t::DELTA_WINDOW;

            float sum = 0.f;
            for (int i = 0; i < CACHE_SIZE; ++i)
                sum += jitter.delta_cache[i];
            float mean_small = sum / static_cast<float>(CACHE_SIZE);
            float var_small = 0.f;
            for (int i = 0; i < CACHE_SIZE; ++i) {
                float d = jitter.delta_cache[i] - mean_small;
                var_small += d * d;
            }
            jitter.delta_variance = var_small / static_cast<float>(CACHE_SIZE);
            analyze_jitter_pattern(jitter);
            analyze_jitter_spectrum(jitter);
            analyze_jitter_crosscorr(jitter);
            analyze_jitter_derivative(jitter);
            analyze_jitter_frequency(jitter);
            analyze_jitter_exponential(jitter);
            update_jitter_fft(jitter, delta);
            analyze_jitter_unified(jitter);
        }

        jitter.yaw_cache[jitter.yaw_cache_offset % YAW_CACHE_SIZE] = current->eye_angles.y;
        jitter.yaw_cache_offset = (jitter.yaw_cache_offset + 1) % YAW_CACHE_SIZE;

        update_jitter_state(jitter);
    }

    inline void prepare_jitter_improved(c_cs_player* player, resolver_info_t& info,
        anim_record_t* current, anim_record_t* last)
    {
        auto& j = info.jitter;

        if (last)
        {
            float delta = math::normalize_yaw(current->eye_angles.y - last->eye_angles.y);
            info.desync_estimator.update(delta, player->velocity().length_2d());

            j.delta_cache[j.cache_offset % CACHE_SIZE] = delta;
            j.cache_offset = (j.cache_offset + 1) % CACHE_SIZE;

            j.delta_history[j.delta_offset % resolver_info_t::jitter_info_t::DELTA_WINDOW] = delta;
            j.delta_offset = (j.delta_offset + 1) % resolver_info_t::jitter_info_t::DELTA_WINDOW;

            float sum = 0.f;
            for (int i = 0; i < CACHE_SIZE; ++i)
                sum += j.delta_cache[i];

            float mean = sum / static_cast<float>(CACHE_SIZE);
            float var = 0.f;
            for (int i = 0; i < CACHE_SIZE; ++i)
            {
                float d = j.delta_cache[i] - mean;
                var += d * d;
            }

            j.delta_variance = var / static_cast<float>(CACHE_SIZE);

            analyze_jitter_pattern(j);
            analyze_jitter_fft(j);
            analyze_jitter_crosscorr(j);
            analyze_jitter_derivative(j);
            analyze_jitter_frequency(j);
            analyze_jitter_exponential(j);
            update_jitter_fft(j, delta);
            analyze_jitter_unified(j);
        }

        j.yaw_cache[j.yaw_cache_offset % YAW_CACHE_SIZE] = current->eye_angles.y;
        j.yaw_cache_offset = (j.yaw_cache_offset + 1) % YAW_CACHE_SIZE;

        update_jitter_state(j);
    }

    inline int movement_based_side(c_cs_player* player, anim_record_t* current)
    {
        if (player->velocity().length_2d() < 1.f)
            return 0;
        vec3_t to_local = math::calc_angle(player->origin(), HACKS->local->origin());
        float velocity_dir = RAD2DEG(std::atan2f(player->velocity().y, player->velocity().x));
        float delta_move = math::normalize_yaw(velocity_dir - to_local.y);
        if (std::fabs(delta_move) > 30.f)
            return delta_move > 0.f ? -1 : 1;

        float delta_eye = math::normalize_yaw(current->eye_angles.y - velocity_dir);
        if (std::fabs(delta_eye) > 40.f)
            return delta_eye > 0.f ? -1 : 1;

        return 0;
    }

    inline int layer_based_side(c_cs_player* player, anim_record_t* current, anim_record_t* last)
    {
        if (!last)
            return 0;

        auto cur_layer = current->layers[ANIMATION_LAYER_MOVEMENT_MOVE];
        auto last_layer = last->layers[ANIMATION_LAYER_MOVEMENT_MOVE];

        if (cur_layer.sequence == last_layer.sequence &&
            cur_layer.playback_rate > last_layer.playback_rate + 0.1f &&
            cur_layer.weight > 0.1f)
        {
            float diff = math::normalize_yaw(current->eye_angles.y - last->eye_angles.y);
            return diff > 0.f ? -1 : 1;
        }

        return 0;
    }

    inline int layer_weight_bias(c_cs_player* player, anim_record_t* rec)
    {
        auto layer = rec->layers[ANIMATION_LAYER_MOVEMENT_MOVE];
        if (layer.weight > 0.55f && layer.playback_rate > 0.01f)
            return layer.cycle > 0.5f ? 1 : -1;
        return 0;
    }

    inline float logistic(float x)
    {
        if (x >= 0.f)
        {
            float e = std::exp(-x);
            return 1.f / (1.f + e);
        }
        float e = std::exp(x);
        return e / (1.f + e);
    }

    inline float compute_final_yaw(const resolver_info_t& info, float base_yaw, float desync_angle)
    {
        const float total =
            info.confidence.jitter * info.accuracy.jitter +
            info.confidence.freestand * info.accuracy.freestand +
            info.confidence.velocity * info.accuracy.velocity +
            info.confidence.foot_delta * info.accuracy.foot_delta +
            info.confidence.brute * info.accuracy.brute;

        if (total <= 0.f)
            return base_yaw;

        const float factor =
            (info.confidence.jitter * info.accuracy.jitter * info.side +
                info.confidence.freestand * info.accuracy.freestand * info.side +
                info.confidence.velocity * info.accuracy.velocity * info.side +
                info.confidence.foot_delta * info.accuracy.foot_delta * info.side +
                info.confidence.brute * info.accuracy.brute * info.side) / total;

        const float weight = logistic(factor * 3.f) * 2.f - 1.f;
        float offset = std::clamp(desync_angle * weight, -desync_angle, desync_angle);
        return math::normalize_yaw(base_yaw + offset);
    }

    inline float final_yaw_softmax(resolver_info_t& info,
        float base_yaw,
        float desync_angle,
        float side_hint)
    {
        float pred[5] = {
            info.confidence.jitter * info.accuracy.jitter * info.side,
            info.confidence.freestand * info.accuracy.freestand * info.side,
            info.confidence.velocity * info.accuracy.velocity * info.side,
            info.confidence.foot_delta * info.accuracy.foot_delta * info.side,
            info.confidence.brute * info.accuracy.brute * info.side
        };

        float sum = 0.f;
        for (float v : pred) sum += std::exp(v);

        float weighted = 0.f;
        for (float v : pred) weighted += v * std::exp(v) / sum;

        float hint_w = logistic(side_hint * 2.f - 1.f);
        float out = desync_angle * (0.8f * weighted + 0.2f * hint_w);

        info.side = out > 0.f ? 1 : -1;
        return math::normalize_yaw(base_yaw + out);
    }

    inline float compute_yaw_weighted(resolver_info_t& info,
        float base_yaw, float desync)
    {
        float predictors[6] =
        {
            info.confidence.jitter * info.accuracy.jitter * info.side,
            info.confidence.freestand * info.accuracy.freestand * info.side,
            info.confidence.velocity * info.accuracy.velocity * info.side,
            info.confidence.foot_delta * info.accuracy.foot_delta * info.side,
            info.confidence.brute * info.accuracy.brute * info.side,
            info.side_prob.best_side() > 0 ? 0.2f : -0.2f
        };

        float sum = 0.f;
        for (float v : predictors)
            sum += std::exp(v);

        float weight = 0.f;
        for (float v : predictors)
            weight += v * std::exp(v) / sum;

        weight = std::clamp(weight, -1.f, 1.f);
        float offset = desync * weight;
        info.side = offset > 0.f ? 1 : -1;

        return math::normalize_yaw(base_yaw + offset);
    }

    inline float compute_final_yaw_hybrid(resolver_info_t& info,
        float base_yaw,
        float desync_ang)
    {
        float pred[5] =
        {
            info.confidence.jitter * info.accuracy.jitter * info.side,
            info.confidence.freestand * info.accuracy.freestand * info.side,
            info.confidence.velocity * info.accuracy.velocity * info.side,
            info.confidence.foot_delta * info.accuracy.foot_delta * info.side,
            info.confidence.brute * info.accuracy.brute * info.side
        };

        float sum = 0.f;
        for (float v : pred) sum += std::exp(v);

        float soft_w = 0.f;
        for (float v : pred) soft_w += v * std::exp(v) / sum;

        float log_w = logistic(soft_w * 2.f) * 2.f - 1.f;
        float side_hint = info.side_prob.best_side() > 0 ? 1.f : -1.f;
        float hint_w = info.side_prob.best_side() > 0 ?
            info.side_prob.prob_right : info.side_prob.prob_left;

        float offset = desync_ang * (0.8f * log_w + 0.2f * side_hint * hint_w);
        info.side = offset > 0.f ? 1 : -1;

        return math::normalize_yaw(base_yaw + offset);
    }

    inline float estimate_desync_range(const resolver_info_t& info, float speed)
    {
        float max_range = info.max_desync;
        float velocity_factor = std::clamp(speed / 150.f, 0.f, 1.f);
        float base = info.desync_estimator.estimate();
        return std::clamp(base * velocity_factor, -max_range, max_range);
    }

    struct yaw_momentum_t { float last_yaw{}; float velocity{}; };
    static yaw_momentum_t yaw_smooth[65]{};

    inline float compute_final_yaw_momentum_ex(resolver_info_t& info,
        float base_yaw,
        float desync_angle,
        int idx,
        bool shot_hit)
    {
        float predicted = final_yaw_softmax(info, base_yaw, desync_angle,
            info.side_prob.best_side() > 0 ?
            info.side_prob.prob_right :
            -info.side_prob.prob_left);

        auto& mom = yaw_smooth[idx];
        float target_v = math::normalize_yaw(predicted - mom.last_yaw);
        mom.velocity = math::lerp(0.5f, mom.velocity, target_v);
        if (shot_hit) mom.velocity *= 0.5f;

        mom.last_yaw = math::normalize_yaw(mom.last_yaw + mom.velocity);
        return mom.last_yaw;
    }

    inline int predict_jitter_phase(const resolver_info_t::jitter_info_t& jitter,
        float current_yaw)
    {
        constexpr int N = YAW_CACHE_SIZE;
        float weight_sum = 0.f;
        float phase = 0.f;

        float w = 1.f;
        const float decay = 0.8f;

        for (int i = 0; i < N - 1; ++i)
        {
            int idx = (jitter.yaw_cache_offset + N - 1 - i) % N;
            int next = (idx + 1) % N;

            float d_cur = math::normalize_yaw(jitter.yaw_cache[idx] - current_yaw);
            float d_next = math::normalize_yaw(jitter.yaw_cache[next] - current_yaw);

            float diff = math::normalize_yaw(d_next - d_cur);

            phase += diff * w;
            weight_sum += w;
            w *= decay;
        }

        phase /= std::max(weight_sum, 0.001f);
        return phase > 0.f ? -1 : 1;
    }

    inline int predict_jitter_phase_advanced(const resolver_info_t::jitter_info_t& j,
        float current_yaw)
    {
        constexpr int N = YAW_CACHE_SIZE;
        float score = 0.f;
        float w = 1.f;
        const float decay = 0.85f;

        for (int i = 0; i < N - 2; ++i)
        {
            int id0 = (j.yaw_cache_offset + N - 1 - i) % N;
            int id1 = (j.yaw_cache_offset + N - 2 - i) % N;

            float d0 = math::normalize_yaw(j.yaw_cache[id0] - current_yaw);
            float d1 = math::normalize_yaw(j.yaw_cache[id1] - current_yaw);
            score += math::normalize_yaw(d0 - d1) * w;
            w *= decay;
        }

        return score > 0.f ? -1 : 1;
    }

    inline int predict_jitter_phase_fir(const resolver_info_t::jitter_info_t& j,
        float current_yaw)
    {
        constexpr int N = YAW_CACHE_SIZE;

        float acc = 0.f;
        float weight = 1.f;
        const float decay = 0.75f;

        for (int i = 0; i < N - 1; ++i)
        {
            int idx0 = (j.yaw_cache_offset + N - 1 - i) % N;
            int idx1 = (j.yaw_cache_offset + N - 2 - i) % N;

            float d0 = math::normalize_yaw(j.yaw_cache[idx0] - current_yaw);
            float d1 = math::normalize_yaw(j.yaw_cache[idx1] - current_yaw);

            acc += math::normalize_yaw(d0 - d1) * weight;
            weight *= decay;
        }

        return acc > 0.f ? -1 : 1;
    }

    struct phase_predictor_t
    {
        float phase{};
        float velocity{};
        float acceleration{};
        float err_est{ 15.f };
        float proc_noise{ 1.f };
        float meas_noise{ 8.f };

        INLINE void reset()
        {
            phase = velocity = acceleration = 0.f;
            err_est = 15.f;
        }

        INLINE void add(float delta)
        {
            phase += velocity + 0.5f * acceleration;
            err_est += proc_noise;

            float gain = err_est / (err_est + meas_noise);
            float resid = delta - phase;

            phase += gain * resid;
            velocity += gain * resid * 0.5f;
            acceleration = math::lerp(0.3f, acceleration, resid);

            err_est *= (1.f - gain);
        }

        INLINE int sign() const { return phase > 0.f ? -1 : 1; }
    };

    inline int predict_jitter_phase_kalman_adv(resolver_info_t::jitter_info_t& j,
        int player_idx)
    {
        static phase_predictor_t filters[65];
        auto& f = filters[player_idx];

        int cur = (j.yaw_cache_offset + YAW_CACHE_SIZE - 1) % YAW_CACHE_SIZE;
        int prev = (cur + YAW_CACHE_SIZE - 1) % YAW_CACHE_SIZE;

        float delta = math::normalize_yaw(j.yaw_cache[cur] - j.yaw_cache[prev]);
        f.add(delta);

        return f.phase > 0.f ? -1 : 1;
    }

    inline int predict_desync_swing_window(const resolver_info_t::jitter_info_t& j,
        c_cs_player* player)
    {
        constexpr int N = 4;
        float sum = 0.f;
        float w = 1.f;
        const float decay = 0.7f;

        int off = j.yaw_cache_offset;
        vec3_t spine = player->get_hitbox_position(HITBOX_CHEST, nullptr);
        vec3_t pelvis = player->get_hitbox_position(HITBOX_PELVIS, nullptr);
        vec3_t dir = spine - pelvis;
        if (dir.length_sqr() < 1e-3f)
            return 0;

        math::vector_angles(dir, dir);

        for (int i = 0; i < N; ++i)
        {
            int idx0 = (off + YAW_CACHE_SIZE - 1 - i) % YAW_CACHE_SIZE;
            int idx1 = (off + YAW_CACHE_SIZE - 2 - i) % YAW_CACHE_SIZE;

            float d0 = math::normalize_yaw(j.yaw_cache[idx0] - dir.y);
            float d1 = math::normalize_yaw(j.yaw_cache[idx1] - dir.y);
            float delta = math::normalize_yaw(d0 - d1);

            sum += delta * w;
            w *= decay;
        }
        return sum > 0.f ? -1 : 1;
    }

    inline int bone_orientation_side(c_cs_player* player)
    {
        vec3_t spine = player->get_hitbox_position(HITBOX_CHEST, nullptr);
        vec3_t head = player->get_hitbox_position(HITBOX_HEAD, nullptr);

        vec3_t dir = head - spine;
        if (dir.length_sqr() < 1e-3f)
            return 0;

        math::vector_angles(dir, dir);
        vec3_t to_local = math::calc_angle(player->origin(), HACKS->local->origin());
        float diff = math::normalize_yaw(dir.y - to_local.y);

        return diff > 0.f ? -1 : 1;
    }

    inline float estimate_true_yaw(c_cs_player* player, const resolver_info_t& info)
    {
        auto hdr = player->get_studio_hdr();
        if (!hdr)
            return player->eye_angles().y;

        vec3_t spine = player->get_hitbox_position(HITBOX_CHEST, nullptr);
        vec3_t pelvis = player->get_hitbox_position(HITBOX_PELVIS, nullptr);
        vec3_t spine_dir = spine - pelvis;
        if (spine_dir.length_sqr() < 1e-3f)
            return player->eye_angles().y;

        math::vector_angles(spine_dir, spine_dir);

        float move_yaw = RAD2DEG(std::atan2f(player->velocity().y, player->velocity().x));
        float blend = std::clamp(player->velocity().length_2d() / 100.f, 0.f, 1.f);
        float yaw = math::lerp(blend, spine_dir.y, move_yaw);

        float side_sign = info.side_prob.best_side() > 0 ? -1.f : 1.f;
        yaw += side_sign * info.max_desync * 0.5f;

        return math::normalize_yaw(yaw);
    }

    inline void update_from_last_reliable(c_cs_player* player, resolver_info_t& info, anim_record_t* current)
    {
        if (!current)
            return;

        float diff = math::normalize_yaw(current->last_reliable_angle.y - player->eye_angles().y);
        if (std::fabs(diff) > 35.f) {
            info.side = diff > 0.f ? -1 : 1;
            info.mode = "last_reliable";
            info.resolved = true;
            info.confidence.jitter = 0.75f;
        }
    }

    inline void try_lock_side(resolver_info_t& info, int predicted_side)
    {
        if (info.locked_side == 0 || HACKS->global_vars->curtime > info.lock_time + 1.0f) {
            info.locked_side = predicted_side;
            info.lock_time = HACKS->global_vars->curtime;
        }

        info.side = info.locked_side;
    }

    inline void update_freestanding_extended(c_cs_player* player,
        resolver_info_t& info,
        anim_record_t* current)
    {
        vec3_t start = HACKS->local->get_eye_position();
        float yaw = current->eye_angles.y;

        auto make_pos = [&](float ang)
            {
                return player->get_eye_position() +
                    vec3_t(std::cos(DEG2RAD(yaw + ang)), std::sin(DEG2RAD(yaw + ang)), 0.f) * 32.f;
            };

        vec3_t angles[4] = { make_pos(-30.f), make_pos(30.f), make_pos(-60.f), make_pos(60.f) };
        c_game_trace traces[4]{};

        c_trace_filter filter{};
        filter.skip = player;

        HACKS->engine_trace->trace_ray(ray_t(start, angles[0]), MASK_SHOT, &filter, &traces[0]);
        HACKS->engine_trace->trace_ray(ray_t(start, angles[1]), MASK_SHOT, &filter, &traces[1]);
        HACKS->engine_trace->trace_ray(ray_t(start, angles[2]), MASK_SHOT, &filter, &traces[2]);
        HACKS->engine_trace->trace_ray(ray_t(start, angles[3]), MASK_SHOT, &filter, &traces[3]);

        float left_frac = (traces[0].fraction + traces[2].fraction) * 0.5f;
        float right_frac = (traces[1].fraction + traces[3].fraction) * 0.5f;

        info.freestanding.left_fraction = left_frac;
        info.freestanding.right_fraction = right_frac;
        info.freestanding.side = left_frac < right_frac ? 1 : -1;
        info.freestanding.update_time = HACKS->global_vars->curtime;
        info.freestanding.updated = true;
    }

    inline int geometry_side_hint(c_cs_player* player, float eye_yaw)
    {
        vec3_t origin = player->get_eye_position();
        const float angles[4] = { 45.f, -45.f, 90.f, -90.f };
        float scores[2]{};

        c_trace_filter filter{};
        filter.skip = player;

        for (int i = 0; i < 4; ++i)
        {
            vec3_t dir(std::cos(DEG2RAD(eye_yaw + angles[i])),
                std::sin(DEG2RAD(eye_yaw + angles[i])), 0.f);
            c_game_trace tr{};
            HACKS->engine_trace->trace_ray(ray_t(origin, origin + dir * 48.f), MASK_SHOT, &filter, &tr);
            if (angles[i] > 0)
                scores[1] += tr.fraction;
            else
                scores[0] += tr.fraction;
        }

        return scores[0] < scores[1] ? 1 : -1;
    }

    inline int geometry_side_hint_extended(c_cs_player* player, float eye_yaw)
    {
        vec3_t origin = player->get_eye_position();
        const float angles[6] = { 30.f, -30.f, 60.f, -60.f, 90.f, -90.f };
        float scores[2]{};

        c_trace_filter filter{};
        filter.skip = player;

        for (int i = 0; i < 6; ++i)
        {
            vec3_t dir(std::cos(DEG2RAD(eye_yaw + angles[i])),
                std::sin(DEG2RAD(eye_yaw + angles[i])), 0.f);
            c_game_trace tr{};
            HACKS->engine_trace->trace_ray(ray_t(origin, origin + dir * 48.f),
                MASK_SHOT, &filter, &tr);
            if (angles[i] > 0)
                scores[1] += tr.fraction;
            else
                scores[0] += tr.fraction;
        }

        return scores[0] < scores[1] ? 1 : -1;
    }

    inline int geometry_side_hint_smooth(c_cs_player* player, float eye_yaw)
    {
        vec3_t origin = player->get_eye_position();
        const float angs[6] = { 30.f, -30.f, 60.f, -60.f, 90.f, -90.f };
        float score[2]{};

        c_trace_filter flt{};
        flt.skip = player;

        for (int i = 0; i < 6; ++i)
        {
            float ang = eye_yaw + angs[i];
            vec3_t dir(std::cos(DEG2RAD(ang)), std::sin(DEG2RAD(ang)), 0.f);
            c_game_trace tr{};
            HACKS->engine_trace->trace_ray(ray_t(origin, origin + dir * 48.f),
                MASK_SHOT, &flt, &tr);
            float w = logistic(angs[i] * 0.1f);
            if (angs[i] > 0)
                score[1] += tr.fraction * w;
            else
                score[0] += tr.fraction * w;
        }

        return score[0] > score[1] ? -1 : 1;
    }

    inline int geometry_side_hint_distance(c_cs_player* player, float eye_yaw)
    {
        vec3_t origin = player->get_eye_position();
        float dist = (HACKS->local->origin() - player->origin()).length();
        float ray_len = std::clamp(dist * 0.5f, 24.f, 96.f);

        const float angs[6] = { 30.f, -30.f, 60.f, -60.f, 90.f, -90.f };
        float score[2]{};

        c_trace_filter flt{};
        flt.skip = player;

        for (int i = 0; i < 6; ++i)
        {
            float ang = eye_yaw + angs[i];
            vec3_t dir(std::cos(DEG2RAD(ang)), std::sin(DEG2RAD(ang)), 0.f);
            c_game_trace tr{};
            HACKS->engine_trace->trace_ray(ray_t(origin, origin + dir * ray_len),
                MASK_SHOT, &flt, &tr);
            float w = logistic(angs[i] * 0.1f);
            if (angs[i] > 0)
                score[1] += tr.fraction * w;
            else
                score[0] += tr.fraction * w;
        }

        return score[0] > score[1] ? -1 : 1;
    }

    inline float choose_brute_offset(resolver_info_t& info, float time_now)
    {
        float offsets[7] = {
            0.f,
            info.max_desync * 0.5f,
            -info.max_desync * 0.5f,
            info.max_desync,
            -info.max_desync,
            info.max_desync * 0.75f,
            -info.max_desync * 0.75f
        };

        for (int i = 0; i < 7; ++i)
        {
            info.brute_hits[i] = std::max(0, info.brute_hits[i] - 1);
        }

        float score_sum = 0.f;
        float scores[7]{};
        for (int i = 0; i < 7; ++i)
        {
            float side = offsets[i] > 0.f ? 1.f : -1.f;
            float base = static_cast<float>(info.brute_hits[i]) + 0.1f;

            if (side == info.side_prob.best_side())
                base *= 1.4f;

            scores[i] = std::exp(base);
            score_sum += scores[i];
        }

        float r = math::random_float(0.f, score_sum);
        int idx = 0;
        for (; idx < 7; ++idx)
        {
            r -= scores[idx];
            if (r <= 0.f)
                break;
        }
        idx = std::clamp(idx, 0, 6);

        std::copy(std::begin(offsets), std::end(offsets), info.brute_offsets);
        info.brute_cycle = idx;
        return offsets[idx];
    }

    inline float adaptive_brute_offset(resolver_info_t& info, float last_miss_angle)
    {
        float base = info.max_desync;
        float side_p = info.side_prob.best_side() > 0 ? info.side_prob.prob_right
            : info.side_prob.prob_left;
        base *= 0.5f + 0.5f * side_p;

        float dynamic = base * (1.f - info.accuracy.brute * 0.25f);
        if (std::fabs(last_miss_angle) > 5.f)
            dynamic += last_miss_angle * 0.2f;

        info.brute_cycle = (info.brute_cycle + 1) % 7;
        return std::clamp(dynamic, -info.max_desync, info.max_desync);
    }

    inline float next_brute_offset(resolver_info_t& info)
    {
        int predicted_step = info.brute_pattern.predict_next(7);
        float offset = info.brute_offsets[predicted_step];

        if (info.side_prob.best_side() > 0)
            offset = -offset;

        info.brute_cycle = predicted_step;
        return offset;
    }

    inline void brute_resolve_predict(c_cs_player* player, resolver_info_t& info,
        anim_record_t* current)
    {
        auto state = player->animstate();
        if (!state)
            return;

        float angle = next_brute_offset(info);
        state->abs_yaw = math::normalize_yaw(current->eye_angles.y + angle);
    }

    inline void brute_resolve_cycle(c_cs_player* player, resolver_info_t& info, anim_record_t* current)
    {
        auto state = player->animstate();
        if (!state)
            return;

        float angle = choose_brute_offset(info, HACKS->global_vars->curtime);
        state->abs_yaw = math::normalize_yaw(current->eye_angles.y + angle);
    }

    inline void prepare_bones_with_resolver(c_cs_player* player, anim_record_t* record)
    {
        auto state = player->animstate();
        if (!state)
            return;

        auto& info = resolver_info[player->index()];
        float desync_delta = state->get_max_rotation();
        float predicted_yaw = record->eye_angles.y;

        if (info.resolved)
            predicted_yaw = math::normalize_yaw(predicted_yaw + desync_delta * info.side);

        state->abs_yaw = predicted_yaw;
    }

    inline float angle_miss_distance(float fired, float impact)
    {
        return std::fabsf(math::normalize_yaw(fired - impact));
    }

    static INLINE void update_side_prob_enhanced(resolver_info_t::side_probability_t& p,
        bool hit, int predicted_side,
        float velocity, int choke,
        float curtime, float& last_time)
    {
        float dt = curtime - last_time;

        float choke_scale = std::clamp(static_cast<float>(choke) / 10.f, 0.f, 1.f);
        float decay = std::exp(-dt * (0.5f + 0.5f * choke_scale));

        p.prob_left = p.prob_left * decay + 0.25f * (1.f - decay);
        p.prob_right = p.prob_right * decay + 0.25f * (1.f - decay);

        float dyn = std::clamp((velocity + 1.f) / 200.f, 0.f, 1.f);
        float alpha = hit ? 0.7f + 0.3f * dyn : 0.3f + 0.2f * dyn;

        if (predicted_side > 0)
            p.prob_right = p.prob_right * alpha + p.prob_left * (1.f - alpha);
        else
            p.prob_left = p.prob_left * alpha + p.prob_right * (1.f - alpha);

        p.normalize();
        last_time = curtime;
    }

    inline void on_shot_result(c_cs_player* player, bool hit, int step,
        float fired_yaw, float impact_yaw)
    {
        auto& info = resolver_info[player->index()];
        if (!hit)
        {
            if (step >= 0 && step < 7)
                info.brute_hits[step] = std::max(0, info.brute_hits[step] - 1);

            info.brute_cycle = (info.brute_cycle + 1) % 7;
        }
        else
        {
            if (step >= 0 && step < 7)
            {
                info.brute_hits[step]++;
                info.brute_pattern.update(step);
            }

            info.brute_cycle = step;
        }

        int predicted_side = step < 7 ? (info.brute_offsets[step] > 0.f ? 1 : -1) : info.side;
        float vel = player->velocity().length_2d();
        float est_desync = info.desync_estimator.estimate();
        update_side_prob_enhanced(info.side_prob, hit, predicted_side,
            vel, HACKS->client_state->choked_commands,
            HACKS->global_vars->curtime, info.side_prob_last_time);

        auto adjust = [&](float& v, float factor)
            {
                v += hit ? 0.05f * factor : -0.05f * factor;
            };

        float miss_factor = hit ? 1.f
            : math::reval_map_clamped(angle_miss_distance(fired_yaw, impact_yaw),
                0.f, 45.f, 1.f, 0.2f);

        const std::string& mode = info.mode;
        if (mode.find("jitter") != std::string::npos)
            adjust(info.accuracy.jitter, miss_factor);
        else if (mode.find("foot_yaw") != std::string::npos)
            adjust(info.accuracy.foot_delta, miss_factor);
        else if (mode.find("freestanding") != std::string::npos)
            adjust(info.accuracy.freestand, miss_factor);
        else if (mode.find("velocity") != std::string::npos || mode.find("movement") != std::string::npos || mode.find("anim") != std::string::npos)
            adjust(info.accuracy.velocity, miss_factor);
        else if (mode.find("brute") != std::string::npos)
            adjust(info.accuracy.brute, miss_factor);

        info.accuracy.clamp();
    }

    inline void prepare_side(c_cs_player* player, anim_record_t* current, anim_record_t* last)
    {
        auto& info = resolver_info[player->index()];
        if (!HACKS->weapon_info || !HACKS->local || !HACKS->local->is_alive() || player->is_bot() || !g_cfg.rage.resolver)
        {
            if (info.resolved)
                info.reset();

            return;
        }

        auto state = player->animstate();
        if (!state)
        {
            if (info.resolved)
                info.reset();

            return;
        }

        auto hdr = player->get_studio_hdr();
        if (!hdr)
            return;

        info.max_desync = state->get_max_rotation();
        info.brute_offsets[0] = 0.f;
        info.brute_offsets[1] = info.max_desync * 0.5f;
        info.brute_offsets[2] = -info.max_desync * 0.5f;
        info.brute_offsets[3] = info.max_desync;
        info.brute_offsets[4] = -info.max_desync;
        info.brute_offsets[5] = info.max_desync * 0.75f;
        info.brute_offsets[6] = -info.max_desync * 0.75f;

        if (current->choke < 2)
            info.add_legit_ticks();
        else
            info.add_fake_ticks();

        if (info.is_legit())
        {
            info.resolved = false;
            info.mode = "no fake";
            return;
        }

        prepare_jitter(player, info, current, last);

        auto& conf = info.confidence;
        conf.reset();

        update_from_last_reliable(player, info, current);

        int best_side = 0;
        float best_score = 0.f;
        const char* best_mode = "";

        auto update_best = [&](float score, int side, const char* mode, float& conf_field)
            {
                conf_field = score;
                if (score > best_score)
                {
                    best_score = score;
                    best_side = side;
                    best_mode = mode;
                }
            };

        auto& jitter = info.jitter;
        if (jitter.is_jitter)
        {
            int jitter_side = predict_jitter_phase_kalman_adv(jitter, player->index());
            float w = jitter.high_freq_jitter ? 1.2f : 1.f;
            update_best(w, jitter_side, "jitter_side", conf.jitter);

            if (jitter.frequency > 0)
            {
                int swing = predict_desync_swing_window(jitter, player);
                update_best(1.f, swing, "bone_swing", conf.jitter);
            }
        }
        else {
            float est = info.desync_estimator.estimate();
            if (std::fabs(est) > 3.f)
                update_best(0.7f, est > 0.f ? -1 : 1, "weighted", conf.jitter);
        }

        float desync_delta = math::normalize_yaw(state->abs_yaw - current->eye_angles.y);
        float max_desync = current->choke < 2 ? state->get_max_rotation() : 60.f;
        if (std::fabs(desync_delta) <= max_desync)
            update_best(0.8f, desync_delta > 0.f ? -1 : 1, "foot_yaw delta", conf.foot_delta);

        auto& free = info.freestanding;
        if (!free.updated || HACKS->global_vars->curtime - free.update_time > 0.8f)
            update_freestanding_extended(player, info, current);

        if (free.updated)
            update_best(0.5f, free.side, "freestanding", conf.freestand);

        int hint_side = geometry_side_hint_distance(player, current->eye_angles.y);
        update_best(0.25f, hint_side, "geometry", conf.freestand);

        vec3_t at_target = math::calc_angle(player->origin(), HACKS->local->origin());
        float move_dir = RAD2DEG(std::atan2f(player->velocity().y, player->velocity().x));
        float diff_move = math::normalize_yaw(move_dir - at_target.y);
        if (std::fabs(diff_move) > 45.f)
            update_best(0.3f, diff_move > 0.f ? -1 : 1, "velocity", conf.velocity);

        int move_side = movement_based_side(player, current);
        if (move_side != 0)
            update_best(0.4f, move_side, "movement dir", conf.velocity);

        int anim_side = layer_based_side(player, current, last);
        if (anim_side != 0)
            update_best(0.6f, anim_side, "anim layer", conf.freestand);
        int bias = layer_weight_bias(player, current);
        if (bias != 0)
            update_best(0.35f, bias, "layer bias", conf.freestand);

        auto& misses = RAGEBOT->missed_shots[player->index()];
        if (misses > 0)
        {
            if (info.brute_step >= 7)
                info.brute_step = 0;

            int side = info.brute_offsets[info.brute_step] > 0.f ? 1 : -1;
            update_best(0.1f, side, "brute", conf.brute);
            info.brute_step++;
        }
        else
            info.brute_step = 0;

        try_lock_side(info, best_side);
        info.resolved = best_score > 0.f;
        info.mode = best_mode;
    }

    inline void apply_side(c_cs_player* player, anim_record_t* current, int choke)
    {
        auto& info = resolver_info[player->index()];
        if (!HACKS->weapon_info || !HACKS->local || !HACKS->local->is_alive() || player->is_teammate(false))
            return;

        auto state = player->animstate();
        if (!state)
            return;

        float vel = player->velocity().length_2d();
        float adaptive = estimate_desync_range(info, vel);
        float final_yaw = player->eye_angles().y;
        if (info.resolved)
            final_yaw = compute_final_yaw_momentum_ex(info, final_yaw, std::fabs(adaptive), player->index(), false);
        state->abs_yaw = math::normalize_yaw(final_yaw);
    }

    inline void prepare_side_improved(c_cs_player* player,
        anim_record_t* current, anim_record_t* last)
    {
        auto& info = resolver_info[player->index()];
        if (!HACKS->weapon_info || !HACKS->local || !HACKS->local->is_alive() || player->is_bot())
        {
            if (info.resolved)
                info.reset();
            return;
        }

        auto state = player->animstate();
        if (!state)
        {
            if (info.resolved)
                info.reset();
            return;
        }

        info.max_desync = state->get_max_rotation();
        prepare_jitter_improved(player, info, current, last);

        auto& conf = info.confidence;
        conf.reset();
        update_from_last_reliable(player, info, current);

        int best_side = 0;
        float best_score = 0.f;
        const char* best_mode = "";

        auto update_best =
            [&](float score, int side, const char* mode, float& field)
            {
                field = score;
                if (score > best_score)
                {
                    best_score = score;
                    best_side = side;
                    best_mode = mode;
                }
            };

        if (info.jitter.is_jitter)
        {
            int phase = predict_jitter_phase_kalman_adv(info.jitter, player->index());
            update_best(1.f, phase, "jitter_phase", conf.jitter);

            int bone_side = bone_orientation_side(player);
            if (bone_side != 0)
                update_best(0.7f, bone_side, "bone_side", conf.freestand);
        }
        else
        {
            float est = info.desync_estimator.estimate();
            if (std::fabs(est) > 5.f)
                update_best(0.5f, est > 0.f ? -1 : 1, "kalman", conf.jitter);
        }

        int hint = geometry_side_hint_distance(player, current->eye_angles.y);
        update_best(0.25f, hint, "geometry", conf.freestand);
        int bias = layer_weight_bias(player, current);
        if (bias != 0)
            update_best(0.35f, bias, "layer bias", conf.freestand);

        try_lock_side(info, best_side);
        info.resolved = best_score > 0.f;
        info.mode = best_mode;
    }

    inline void apply_side_improved(c_cs_player* player,
        anim_record_t* current, int choke)
    {
        auto& info = resolver_info[player->index()];
        if (!HACKS->weapon_info || !HACKS->local || !HACKS->local->is_alive() ||
            player->is_teammate(false))
            return;

        auto state = player->animstate();
        if (!state)
            return;

        float vel = player->velocity().length_2d();
        float adaptive = estimate_desync_range(info, vel);
        float final_yaw = player->eye_angles().y;

        if (info.resolved)
        {
            float predicted = estimate_true_yaw(player, info);
            final_yaw = final_yaw_softmax(info, predicted, std::fabs(adaptive),
                info.side_prob.best_side() > 0 ?
                info.side_prob.prob_right :
                -info.side_prob.prob_left);
        }

        state->abs_yaw = math::normalize_yaw(final_yaw);
    }

    inline void brute_resolve_adaptive(c_cs_player* player,
        resolver_info_t& info,
        anim_record_t* current,
        float last_miss)
    {
        auto state = player->animstate();
        if (!state)
            return;

        float angle = adaptive_brute_offset(info, last_miss);
        state->abs_yaw = math::normalize_yaw(current->eye_angles.y + angle);
    }

    inline float compute_dynamic_brute_offset(resolver_info_t& info,
        float last_miss_delta)
    {
        static const float base_table[9] =
        {
            0.f, 0.25f, -0.25f, 0.5f, -0.5f,
            0.75f, -0.75f, 1.f, -1.f
        };

        int idx = info.brute_cycle % 9;
        float base = base_table[idx] * info.max_desync;

        float side_bias = info.side_prob.best_side() > 0 ? -1.f : 1.f;
        float prob = info.side_prob.best_side() > 0 ? info.side_prob.prob_right
            : info.side_prob.prob_left;
        base *= 0.5f + 0.5f * prob;

        base += last_miss_delta * 0.1f * side_bias;

        info.brute_cycle = (info.brute_cycle + 1) % 9;
        return std::clamp(base, -info.max_desync, info.max_desync);
    }

    static INLINE float compute_brute_gradient(resolver_info_t& info,
        float last_miss_delta)
    {
        static const float step_table[5] =
        { 0.f, 0.25f, -0.25f, 0.5f, -0.5f };

        int idx = info.brute_cycle % 5;
        float base = step_table[idx] * info.max_desync;

        float grad = last_miss_delta * 0.2f;
        base += grad;

        float bias = info.side_prob.best_side() > 0 ?
            info.side_prob.prob_right : -info.side_prob.prob_left;
        base += bias * 0.1f * info.max_desync;

        info.brute_cycle = (info.brute_cycle + 1) % 5;
        return std::clamp(base, -info.max_desync, info.max_desync);
    }

    inline float update_brute_gradient(resolver_info_t& info, float miss_delta)
    {
        static const float steps[] = { 0.f, 0.25f, -0.25f, 0.5f, -0.5f };
        int idx = info.brute_cycle % 5;

        float base = steps[idx] * info.max_desync;
        float miss_factor = logistic(miss_delta * 0.02f) * 2.f - 1.f;

        base += miss_factor * info.max_desync * 0.25f;

        float hit_bias = info.side_prob.best_side() > 0 ?
            info.side_prob.prob_right : -info.side_prob.prob_left;

        base += hit_bias * info.max_desync * 0.1f;

        info.brute_cycle = (info.brute_cycle + 1) % 5;
        return std::clamp(base, -info.max_desync, info.max_desync);
    }

    inline void brute_resolve_adaptive_grad(c_cs_player* player,
        resolver_info_t& info,
        anim_record_t* rec,
        float last_miss)
    {
        auto state = player->animstate();
        if (!state)
            return;

        float off = update_brute_gradient(info, last_miss);
        state->abs_yaw = math::normalize_yaw(rec->eye_angles.y + off);
    }

    inline void brute_resolve_gradient(c_cs_player* player,
        resolver_info_t& info,
        anim_record_t* current,
        float miss_delta)
    {
        auto state = player->animstate();
        if (!state)
            return;

        float offset = compute_brute_gradient(info, miss_delta);
        state->abs_yaw =
            math::normalize_yaw(current->eye_angles.y + offset);
    }

    inline void brute_resolve_dynamic(c_cs_player* player,
        resolver_info_t& info,
        anim_record_t* current,
        float last_miss_delta)
    {
        auto state = player->animstate();
        if (!state)
            return;

        float angle = compute_dynamic_brute_offset(info, last_miss_delta);
        state->abs_yaw = math::normalize_yaw(current->eye_angles.y + angle);
    }
}