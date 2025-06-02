#include "globals.hpp"
#include "resolver.hpp"
#include "animations.hpp"
#include "server_bones.hpp"
#include "ragebot.hpp"
#include "defines.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <mutex>

void resolver_analytics::log_tick(const tick_data& data) {
    std::lock_guard<std::mutex> lock(data_mutex);
    data_log.push_back(data);
}

void resolver_analytics::export_csv(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    std::lock_guard<std::mutex> lock(data_mutex);

    file << "timestamp,player_id,predicted_yaw,confidence,chosen_side,hit\n";

    for (const auto& entry : data_log) {
        file << entry.timestamp << ","
            << entry.player_id << ","
            << entry.predicted_yaw << ","
            << entry.confidence << ","
            << entry.chosen_side << ","
            << (entry.hit ? 1 : 0) << "\n";
    }
}

namespace resolver {
    resolver_context g_context{};
    resolver_analytics g_analytics{};

    namespace {
        constexpr int WN = DELTA_WINDOW;
        alignas(32) float ψ_re[WN]{};
        alignas(32) float ψ_im[WN]{};

        struct _cwti {
            _cwti() {
                for (int n = 0; n < WN; ++n) {
                    float t = (float(n) - WN / 2) / 4.f;
                    float g = std::exp(-t * t / 2.f);
                    ψ_re[n] = g * std::cos(6.f * t);
                    ψ_im[n] = g * std::sin(6.f * t);
                }
            }
        } _cwti_;

        INLINE void sincos_deg(int d, float& s, float& c) {
            d = (d % 360 + 360) % 360;
            s = trig_tables::sin_table[d];
            c = trig_tables::cos_table[d];
        }

        INLINE float norm_angle(float a) {
            a = std::fmod(a + 180.f, 360.f);
            if (a < 0) a += 360.f;
            return a - 180.f;
        }

        INLINE float fast_sin_deg(float a) {
            int d = int(a);
            if (d >= 0 && d < TABLE_SIZE) {
                return trig_tables::sin_table[d];
            }
            float s, c;
            sincos_deg(d, s, c);
            return s;
        }

        INLINE float fast_cos_deg(float a) {
            int d = int(a);
            if (d >= 0 && d < TABLE_SIZE) {
                return trig_tables::cos_table[d];
            }
            float s, c;
            sincos_deg(d, s, c);
            return c;
        }
    }

    void update_context(int tick_rate, float latency, int choked_commands, bool competitive) {
        g_context.tick_rate = tick_rate;
        g_context.latency = latency;
        g_context.choked_commands = choked_commands;
        g_context.is_competitive = competitive;
    }

    static INLINE void update_jitter_cwt_vectorized(resolver_info_t::jitter_info_t& j, float newest) {
        j.delta_history.push(newest);

#ifdef HAS_AVX512
        __m512 re_acc = _mm512_setzero_ps();
        __m512 im_acc = _mm512_setzero_ps();

        for (int n = 0; n < WN; n += 16) {
            int blk = std::min(16, WN - n);
            alignas(64) float delta_vals[16];
            for (int i = 0; i < blk; ++i) {
                delta_vals[i] = j.delta_history.get_chronological(n + i);
            }
            for (int i = blk; i < 16; ++i) {
                delta_vals[i] = 0.f;
            }

            __m512 delta_vec = _mm512_load_ps(delta_vals);
            __m512 psi_re_vec = _mm512_load_ps(&ψ_re[n]);
            __m512 psi_im_vec = _mm512_load_ps(&ψ_im[n]);

            re_acc = _mm512_fmadd_ps(delta_vec, psi_re_vec, re_acc);
            im_acc = _mm512_fmadd_ps(delta_vec, psi_im_vec, im_acc);
        }

        j.wavelet_re = _mm512_reduce_add_ps(re_acc);
        j.wavelet_im = _mm512_reduce_add_ps(im_acc);

#elif defined(HAS_NEON)
        float32x4_t re_acc = vdupq_n_f32(0.0f);
        float32x4_t im_acc = vdupq_n_f32(0.0f);

        for (int n = 0; n < WN; n += 4) {
            int blk = std::min(4, WN - n);
            float delta_vals[4] = { 0.f, 0.f, 0.f, 0.f };
            for (int i = 0; i < blk; ++i) {
                delta_vals[i] = j.delta_history.get_chronological(n + i);
            }

            float32x4_t delta_vec = vld1q_f32(delta_vals);
            float32x4_t psi_re_vec = vld1q_f32(&ψ_re[n]);
            float32x4_t psi_im_vec = vld1q_f32(&ψ_im[n]);

            re_acc = vmlaq_f32(re_acc, delta_vec, psi_re_vec);
            im_acc = vmlaq_f32(im_acc, delta_vec, psi_im_vec);
        }

        float re_sum[4], im_sum[4];
        vst1q_f32(re_sum, re_acc);
        vst1q_f32(im_sum, im_acc);

        j.wavelet_re = re_sum[0] + re_sum[1] + re_sum[2] + re_sum[3];
        j.wavelet_im = im_sum[0] + im_sum[1] + im_sum[2] + im_sum[3];

#else
        __m256 re_acc = _mm256_setzero_ps();
        __m256 im_acc = _mm256_setzero_ps();

        for (int n = 0; n < WN; n += 8) {
            alignas(32) float delta_vals[8];
            for (int i = 0; i < 8; ++i) {
                delta_vals[i] = j.delta_history.get_chronological(n + i);
            }

            __m256 delta_vec = _mm256_loadu_ps(delta_vals);
            __m256 psi_re_vec = _mm256_load_ps(&ψ_re[n]);
            __m256 psi_im_vec = _mm256_load_ps(&ψ_im[n]);

            re_acc = _mm256_fmadd_ps(delta_vec, psi_re_vec, re_acc);
            im_acc = _mm256_fmadd_ps(delta_vec, psi_im_vec, im_acc);
        }

        __m128 re_low = _mm256_castps256_ps128(re_acc);
        __m128 re_high = _mm256_extractf128_ps(re_acc, 1);
        __m128 re_sum = _mm_add_ps(re_low, re_high);
        re_sum = _mm_hadd_ps(re_sum, re_sum);
        re_sum = _mm_hadd_ps(re_sum, re_sum);
        j.wavelet_re = _mm_cvtss_f32(re_sum);

        __m128 im_low = _mm256_castps256_ps128(im_acc);
        __m128 im_high = _mm256_extractf128_ps(im_acc, 1);
        __m128 im_sum = _mm_add_ps(im_low, im_high);
        im_sum = _mm_hadd_ps(im_sum, im_sum);
        im_sum = _mm_hadd_ps(im_sum, im_sum);
        j.wavelet_im = _mm_cvtss_f32(im_sum);
#endif

        j.strength = std::sqrt(j.wavelet_re * j.wavelet_re + j.wavelet_im * j.wavelet_im);
        j.is_jitter = j.strength > 25.f;
        j.high_freq_jitter = j.strength > 50.f;

        float sum = 0.f;
        for (int i = 0; i < WN; ++i) {
            sum += j.delta_history.get_chronological(i);
        }
        float mean = sum / WN;

        float var = 0.f;
#pragma omp simd reduction(+:var)
        for (int i = 0; i < WN; ++i) {
            float d = j.delta_history.get_chronological(i) - mean;
            var += d * d;
        }
        j.variance = var / WN;

        j.machine.update(j.is_jitter, j.variance);
    }

    static INLINE void update_jitter_incremental_vectorized(resolver_info_t::jitter_info_t& j, float newest) {
        constexpr int N = DELTA_WINDOW;
        float oldMean = j.mean;
        j.mean += (newest - oldMean) / float(N);
        j.m2 += (newest - oldMean) * (newest - j.mean);
        j.variance = j.m2 / (N - 1);

        constexpr float w = 2.f * float(M_PI) * float(resolver_info_t::jitter_info_t::GOERTZEL_K) / N;
        float coeff = 2.f * std::cosf(w);

        float q0 = newest + coeff * j.goertzel_q[1] - j.goertzel_q[2];
        j.goertzel_q[2] = j.goertzel_q[1];
        j.goertzel_q[1] = j.goertzel_q[0];
        j.goertzel_q[0] = q0;

        float power = j.goertzel_q[2] * j.goertzel_q[2] +
            j.goertzel_q[1] * j.goertzel_q[1] -
            coeff * j.goertzel_q[1] * j.goertzel_q[2];
        j.autocorr = power / (N * N);
        j.high_freq_jitter = j.autocorr > 8.f;
        j.is_jitter = j.variance > 9.f || j.high_freq_jitter;
    }

    std::array<float, 16> extract_features(c_cs_player* player, const resolver_info_t& info, anim_record_t* record) {
        auto& j = info.jitter;
        float speed = player->velocity().length_2d();

        return {
            speed * 0.01f,
            j.variance * 0.01f,
            j.strength * 0.001f,
            j.autocorr * 0.1f,
            float(info.fake_ticks) / float(MAX_TICKS),
            float(g_context.choked_commands) * 0.1f,
            info.confidence.jitter,
            info.confidence.foot_delta,
            info.confidence.freestand,
            info.confidence.velocity,
            info.confidence.brute,
            float(info.side),
            info.max_desync * 0.01f,
            float(j.machine.is_jitter()),
            float(g_context.tick_rate) / 128.f,
            g_context.latency * 0.01f
        };
    }

    inline void prepare_jitter(c_cs_player* player, resolver_info_t& info, anim_record_t* c, anim_record_t* l) {
        auto& j = info.jitter;
        if (l) {
            float d = math::normalize_yaw(c->eye_angles.y - l->eye_angles.y);
            d = norm_angle(d);

            j.delta_cache.push(d);

            float sum = 0.f;
            for (int i = 0; i < CACHE_SIZE; ++i) {
                sum += j.delta_cache.get_chronological(i);
            }
            float mean = sum / float(CACHE_SIZE);

            float var = 0.f;
            for (int i = 0; i < CACHE_SIZE; ++i) {
                float dd = j.delta_cache.get_chronological(i) - mean;
                var += dd * dd;
            }
            j.delta_variance = var / float(CACHE_SIZE - 1);

            update_jitter_incremental_vectorized(j, d);

            float tick_interval = 1.f / float(g_context.tick_rate);
            info.ukf->set_dt(tick_interval);
            info.ukf->predict();
            info.ukf->update(c->eye_angles.y);

            info.particle_filter->predict();
            info.particle_filter->update(c->eye_angles.y);
        }

        j.yaw_cache.push(c->eye_angles.y);
    }

    inline int predict_jitter_phase_advanced(resolver_info_t::jitter_info_t& j, int idx) {
        struct phase_hmm_t {
            float p_left{ 0.5f };
            float p_right{ 0.5f };
            float stay{ 0.8f };
            float switch_p{ 0.2f };

            INLINE void update(int m) {
                float eL = m < 0 ? 0.85f : 0.15f;
                float eR = 1.f - eL;
                float nl = eL * (p_left * stay + p_right * switch_p);
                float nr = eR * (p_right * stay + p_left * switch_p);
                float n = nl + nr;
                p_left = std::max(EPSILON, nl / n);
                p_right = std::max(EPSILON, nr / n);
            }

            INLINE int best() const {
                return p_left > p_right ? -1 : 1;
            }
        };

        static std::array<phase_hmm_t, 65> phase_models{};
        auto& hmm = phase_models[idx];

        float current_yaw = j.yaw_cache.get_chronological(0);
        float prev_yaw = j.yaw_cache.get_chronological(1);
        float d = norm_angle(current_yaw - prev_yaw);

        hmm.update(d < 0.f ? -1 : 1);
        return hmm.best();
    }

    inline int movement_based_side(c_cs_player* p, anim_record_t* c) {
        if (p->velocity().length_2d() < 1.f) {
            return 0;
        }

        vec3_t to_local = math::calc_angle(p->origin(), HACKS->local->origin());
        float vel_dir = RAD2DEG(std::atan2f(p->velocity().y, p->velocity().x));
        float delta_move = norm_angle(vel_dir - to_local.y);

        if (std::fabs(delta_move) > 30.f) {
            return delta_move > 0.f ? 1 : -1;
        }

        float delta_eye = norm_angle(c->eye_angles.y - vel_dir);
        if (std::fabs(delta_eye) > 40.f) {
            return delta_eye > 0.f ? 1 : -1;
        }

        return 0;
    }

    inline int layer_based_side(c_cs_player* p, anim_record_t* c, anim_record_t* l) {
        if (!l) return 0;

        auto cur = c->layers[ANIMATION_LAYER_MOVEMENT_MOVE];
        auto last = l->layers[ANIMATION_LAYER_MOVEMENT_MOVE];

        if (cur.sequence == last.sequence &&
            cur.playback_rate > last.playback_rate + 0.1f &&
            cur.weight > 0.1f) {
            float diff = norm_angle(c->eye_angles.y - l->eye_angles.y);
            return diff > 0.f ? 1 : -1;
        }

        return 0;
    }

    inline void update_freestanding_extended(c_cs_player* p, resolver_info_t& info, anim_record_t* c) {
        vec3_t start = HACKS->local->get_eye_position();
        float yaw = c->eye_angles.y;

        auto make_end = [&](float angle) {
            return p->get_eye_position() +
                vec3_t(std::cos(DEG2RAD(yaw + angle)),
                    std::sin(DEG2RAD(yaw + angle)), 0.f) * 32.f;
            };

        std::array<vec3_t, 4> angles = {
            make_end(-30.f), make_end(30.f),
            make_end(-60.f), make_end(60.f)
        };

        std::array<float, 4> fractions{};
        c_trace_filter filter{};
        filter.skip = p;

        for (int i = 0; i < 4; ++i) {
            c_game_trace tr{};
            HACKS->engine_trace->trace_ray(
                ray_t(start, angles[i]), MASK_SHOT, &filter, &tr);
            fractions[i] = tr.fraction;
        }

        float left_frac = (fractions[0] + fractions[2]) * 0.5f;
        float right_frac = (fractions[1] + fractions[3]) * 0.5f;

        info.freestanding.left_fraction = left_frac;
        info.freestanding.right_fraction = right_frac;
        info.freestanding.side =
            std::fabs(left_frac - right_frac) < 1e-3f ? 0 :
            (left_frac < right_frac ? 1 : -1);
        info.freestanding.update_time = HACKS->global_vars->curtime;
        info.freestanding.updated = true;
    }

    inline float compute_adaptive_confidence(resolver_info_t& info) {
        std::array<float, 5> confidences = {
            info.confidence.jitter,
            info.confidence.foot_delta,
            info.confidence.freestand,
            info.confidence.velocity,
            info.confidence.brute
        };

        return info.adaptive_confidence.blend(confidences);
    }

    void prepare_side(c_cs_player* p, anim_record_t* cur, anim_record_t* last) {
        auto& info = resolver_info[p->index()];

        if (!HACKS->weapon_info || !HACKS->local || !HACKS->local->is_alive() /* ||
            p->is_bot() */ || !g_cfg.rage.resolver) {
            if (info.resolved) {
                info.reset();
            }
            return;
        }

        auto st = p->animstate();
        if (!st) {
            if (info.resolved) {
                info.reset();
            }
            return;
        }

        info.state = resolver_state::ANALYZING;
        info.max_desync = st->get_max_rotation();

        info.brute_offsets[0] = 0.f;
        info.brute_offsets[1] = info.max_desync * 0.5f;
        info.brute_offsets[2] = -info.max_desync * 0.5f;
        info.brute_offsets[3] = info.max_desync;
        info.brute_offsets[4] = -info.max_desync;
        info.brute_offsets[5] = info.max_desync * 0.75f;
        info.brute_offsets[6] = -info.max_desync * 0.75f;

        if (cur->choke < 2) {
            info.add_legit_ticks();
        }
        else {
            info.add_fake_ticks();
        }

        if (info.is_legit()) {
            info.resolved = false;
            info.mode = "no fake";
            info.state = resolver_state::IDLE;
            info.brute_step = 0;
            return;
        }

        prepare_jitter(p, info, cur, last);
        update_jitter_cwt_vectorized(info.jitter,
            last ? norm_angle(cur->eye_angles.y - last->eye_angles.y) : 0.f);

        info.skeletal_analyzer.update(p);
        info.latency_compensator.update(g_context);

        auto& conf = info.confidence;
        conf.reset();

        auto features = extract_features(p, info, cur);
        int neural_side = info.neural_net->predict(features);

        float neural_confidence = 0.8f;
        conf.jitter = std::max(conf.jitter, neural_confidence);

        int best_side = neural_side;
        float best_score = neural_confidence;
        const char* best_mode = "neural";

        auto update_best = [&](float score, int side, const char* mode, float& confidence) {
            confidence = std::max(confidence, score);
            if (score > best_score) {
                best_score = score;
                best_side = side;
                best_mode = mode;
            }
            };

        auto& j = info.jitter;
        if (j.is_jitter) {
            int jitter_side = predict_jitter_phase_advanced(j, p->index());
            float weight = j.high_freq_jitter ? 1.2f : 1.f;
            update_best(weight, jitter_side, "jitter_phase", conf.jitter);

            if (j.wavelet_im != 0.f) {
                int cwt_side = j.wavelet_im > 0.f ? 1 : -1;
                update_best(1.1f, cwt_side, "cwt", conf.jitter);
            }
        }

        float ukf_estimate = info.ukf->estimate();
        if (std::fabs(ukf_estimate) > 4.f) {
            int ukf_side = ukf_estimate > 0.f ? 1 : -1;
            update_best(0.9f, ukf_side, "ukf", conf.jitter);
        }

        float pf_estimate = info.particle_filter->estimate();
        if (std::fabs(pf_estimate) > 5.f) {
            int pf_side = pf_estimate > 0.f ? 1 : -1;
            update_best(0.85f, pf_side, "particle", conf.jitter);
        }

        int skeletal_side = info.skeletal_analyzer.predict_side();
        if (skeletal_side != 0) {
            update_best(0.7f, skeletal_side, "skeletal", conf.foot_delta);
        }

        float desync_delta = norm_angle(st->abs_yaw - cur->eye_angles.y);
        float max_delta = cur->choke < 2 ? st->get_max_rotation() : 60.f;
        if (std::fabs(desync_delta) <= max_delta) {
            update_best(0.8f, desync_delta > 0.f ? 1 : -1, "foot_delta", conf.foot_delta);
        }

        auto& fr = info.freestanding;
        if (!fr.updated || HACKS->global_vars->curtime - fr.update_time > 0.8f) {
            update_freestanding_extended(p, info, cur);
        }
        if (fr.updated) {
            update_best(0.6f, fr.side, "freestanding", conf.freestand);
        }

        int movement_side = movement_based_side(p, cur);
        if (movement_side) {
            update_best(0.5f, movement_side, "movement", conf.velocity);
        }

        int layer_side = layer_based_side(p, cur, last);
        if (layer_side) {
            update_best(0.6f, layer_side, "layer", conf.velocity);
        }

        auto& missed = RAGEBOT->missed_shots[p->index()];
        if (missed > 0) {
            info.state = resolver_state::BRUTE_FORCING;

            if (info.brute_step >= 7) {
                info.brute_step = 0;
            }

            auto ctx = g_context.make_feature_vector();
            std::array<float, 8> bandit_context = {
                ctx[0], ctx[1], ctx[2], ctx[3],
                p->velocity().length_2d() * 0.01f,
                info.jitter.variance * 0.01f,
                float(info.brute_step) / 7.f,
                float(missed) * 0.1f
            };

            int bandit_choice = info.bandit.select(bandit_context);
            float offset = info.brute_offsets[bandit_choice];
            int brute_side = offset > 0.f ? 1 : -1;

            update_best(0.3f, brute_side, "bandit_brute", conf.brute);
            info.brute_step = bandit_choice;
        }

        if (info.locked_side == 0 || HACKS->global_vars->curtime > info.lock_time + 1.f) {
            info.locked_side = best_side;
            info.lock_time = HACKS->global_vars->curtime;
        }

        info.side = info.locked_side;
        info.mode = best_mode;
        info.resolved = best_score > 0.1f;
        info.state = info.resolved ? resolver_state::RESOLVED : resolver_state::ANALYZING;

        g_analytics.log_tick({
            p->index(),
            cur->eye_angles.y + info.max_desync * info.side,
            compute_adaptive_confidence(info),
            info.side,
            false,
            HACKS->global_vars->curtime
            });
    }

    void apply_side(c_cs_player* p, anim_record_t* cur, int choke) {
        auto& info = resolver_info[p->index()];
        auto* st = p->animstate();
        if (!st) return;

        float yaw = cur->eye_angles.y;

        if (info.resolved) {
            if (info.state == resolver_state::BRUTE_FORCING &&
                info.brute_step >= 0 && info.brute_step < 7) {
                float offset = info.brute_offsets[info.brute_step];
                yaw = norm_angle(yaw + offset);
            }
            else {
                float desync_amount = info.max_desync;

                float velocity_factor = std::clamp(p->velocity().length_2d() / 250.f, 0.f, 1.f);
                desync_amount *= (0.6f + 0.4f * velocity_factor);

                float angular_velocity = 0.f;
                if (info.jitter.yaw_cache.index > 1) {
                    float dt = HACKS->global_vars->interval_per_tick;
                    angular_velocity = norm_angle(info.jitter.yaw_cache.get_chronological(0) -
                        info.jitter.yaw_cache.get_chronological(1)) / dt;
                }

                float compensated = info.latency_compensator.compensate_angle(
                    desync_amount * info.side, angular_velocity);

                yaw = norm_angle(yaw + compensated);
            }
        }

        st->abs_yaw = yaw;
    }

    void on_shot_result(c_cs_player* p, bool hit, int step, float fired, float impact) {
        auto& info = resolver_info[p->index()];

        g_analytics.log_tick({
            p->index(),
            fired,
            compute_adaptive_confidence(info),
            info.side,
            hit,
            HACKS->global_vars->curtime
            });

        auto features = extract_features(p, info, nullptr);
        info.neural_net->update(features, info.side, hit);

        if (info.state == resolver_state::BRUTE_FORCING && step >= 0 && step < 7) {
            auto ctx = g_context.make_feature_vector();
            std::array<float, 8> bandit_context = {
                ctx[0], ctx[1], ctx[2], ctx[3],
                p->velocity().length_2d() * 0.01f,
                info.jitter.variance * 0.01f,
                float(step) / 7.f,
                std::fabs(norm_angle(fired - impact)) * 0.01f
            };

            info.bandit.update(step, bandit_context, hit);
        }

        int method_idx = -1;
        const std::string& mode = info.mode;
        if (mode.find("jitter") != std::string::npos || mode.find("cwt") != std::string::npos ||
            mode.find("neural") != std::string::npos || mode.find("ukf") != std::string::npos ||
            mode.find("particle") != std::string::npos) {
            method_idx = 0;
        }
        else if (mode.find("foot") != std::string::npos || mode.find("skeletal") != std::string::npos) {
            method_idx = 1;
        }
        else if (mode.find("freestanding") != std::string::npos) {
            method_idx = 2;
        }
        else if (mode.find("velocity") != std::string::npos || mode.find("movement") != std::string::npos ||
            mode.find("layer") != std::string::npos) {
            method_idx = 3;
        }
        else if (mode.find("brute") != std::string::npos || mode.find("bandit") != std::string::npos) {
            method_idx = 4;
        }

        if (method_idx >= 0) {
            info.adaptive_confidence.update(method_idx, hit);
        }

        if (hit) {
            if (step >= 0 && step < 7) {
                info.brute_hits[step]++;
            }
        }
        else {
            if (step >= 0 && step < 7) {
                info.brute_hits[step] = std::max(0, info.brute_hits[step] - 1);
            }
        }
    }

    void prepare_bones_with_resolver(c_cs_player* p, anim_record_t* r) {
        auto st = p->animstate();
        if (!st) return;

        auto& info = resolver_info[p->index()];
        float yaw = r->eye_angles.y;

        if (info.resolved) {
            if (info.state == resolver_state::BRUTE_FORCING &&
                info.brute_step >= 0 && info.brute_step < 7) {
                yaw = norm_angle(yaw + info.brute_offsets[info.brute_step]);
            }
            else {
                float desync = info.max_desync * info.side;

                float velocity_factor = std::clamp(p->velocity().length_2d() / 250.f, 0.f, 1.f);
                desync *= (0.6f + 0.4f * velocity_factor);

                yaw = norm_angle(yaw + desync);
            }
        }

        st->abs_yaw = yaw;
    }

    void prepare_side_improved(c_cs_player* p, anim_record_t* cur, anim_record_t* last) {
        prepare_side(p, cur, last);
    }

    void apply_side_improved(c_cs_player* p, anim_record_t* cur, int choke) {
        apply_side(p, cur, choke);
    }
}
