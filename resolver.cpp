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

    static INLINE void update_jitter_sliding_window(resolver_info_t::jitter_info_t& j, float newest) {
        float oldest = 0.f;
        if (j.delta_history.count >= DELTA_WINDOW) {
            oldest = j.delta_history.data[j.delta_history.index];
        }

        j.delta_history.push(newest);

        if (j.delta_history.size() < DELTA_WINDOW) {
            j.sum_x = 0.f;
            j.sum_x2 = 0.f;
            for (size_t i = 0; i < j.delta_history.size(); ++i) {
                float val = j.delta_history.get_chronological(i);
                j.sum_x += val;
                j.sum_x2 += val * val;
            }
        }
        else {
            j.sum_x += newest - oldest;
            j.sum_x2 += newest * newest - oldest * oldest;
        }

        size_t n = j.delta_history.size();
        if (n > 1) {
            float mean = j.sum_x / n;
            float s2 = j.sum_x2 - n * mean * mean;
            j.variance = std::max(s2, 0.f) / (n - 1);
        }

        if (j.delta_history.size() >= 2) {
            float x0 = j.delta_history.get_chronological(0);
            float x1 = j.delta_history.get_chronological(1);
            float denom = std::sqrt(x0 * x0 + 1e-6f) * std::sqrt(x1 * x1 + 1e-6f);
            j.autocorr = (x0 * x1) / denom;
        }

        float filtered = j.biquad.process(newest);
        j.high_freq_jitter = std::abs(filtered) > 8.f;

        j.stft.analyze(j.delta_history);
        float max_power = 0.f;
        int max_bin = 0;
        for (int i = 1; i < STFT_SIZE / 2; ++i) {
            float power = j.stft.get_power(i);
            if (power > max_power) {
                max_power = power;
                max_bin = i;
            }
        }

        j.strength = max_power;
        j.frequency = std::clamp(max_bin, 0, STFT_SIZE / 2 - 1);
        j.is_jitter = j.strength > 25.f || j.high_freq_jitter;

        j.machine.update(j.is_jitter, j.variance);
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
            for (size_t i = 0; i < j.delta_cache.size(); ++i) {
                sum += j.delta_cache.get_chronological(i);
            }
            float mean = sum / float(j.delta_cache.size());

            float var = 0.f;
            for (size_t i = 0; i < j.delta_cache.size(); ++i) {
                float dd = j.delta_cache.get_chronological(i) - mean;
                var += dd * dd;
            }
            j.delta_variance = j.delta_cache.size() > 1 ? var / float(j.delta_cache.size() - 1) : 0.f;

            update_jitter_sliding_window(j, d);

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

        if (j.yaw_cache.size() >= 2) {
            float current_yaw = j.yaw_cache.get_chronological(0);
            float prev_yaw = j.yaw_cache.get_chronological(1);
            float d = norm_angle(current_yaw - prev_yaw);

            hmm.update(d < 0.f ? -1 : 1);
        }
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

        if (!HACKS->weapon_info || !HACKS->local || !HACKS->local->is_alive() ||
            p->is_bot() || !g_cfg.rage.resolver) {
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

        info.skeletal_analyzer.update(p);
        info.latency_compensator.update(g_context);

        auto& conf = info.confidence;
        conf.reset();

        auto features = extract_features(p, info, cur);
        int ttt_side = info.ttt_net->predict(features);

        float ttt_confidence = 0.85f;
        conf.jitter = std::max(conf.jitter, ttt_confidence);

        int best_side = ttt_side;
        float best_score = ttt_confidence;
        const char* best_mode = "ttt";

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

            float max_power = 0.f;
            int dominant_bin = 0;
            for (int i = 1; i < STFT_SIZE / 2; ++i) {
                float power = j.stft.get_power(i);
                if (power > max_power) {
                    max_power = power;
                    dominant_bin = i;
                }
            }
            if (max_power > 30.f) {
                int stft_side = dominant_bin > STFT_SIZE / 4 ? 1 : -1;
                update_best(1.1f, stft_side, "stft", conf.jitter);
            }
        }

        float ukf_estimate = info.ukf->estimate();
        float ukf_confidence = info.ukf->confidence();
        if (std::fabs(ukf_estimate) > 4.f && ukf_confidence > 0.5f) {
            int ukf_side = ukf_estimate > 0.f ? 1 : -1;
            update_best(0.9f * ukf_confidence, ukf_side, "sqrt_ukf", conf.jitter);
        }

        float pf_estimate = info.particle_filter->estimate();
        float pf_confidence = info.particle_filter->confidence();
        if (std::fabs(pf_estimate) > 5.f && pf_confidence > 0.5f) {
            int pf_side = pf_estimate > 0.f ? 1 : -1;
            update_best(0.85f * pf_confidence, pf_side, "ar1_particle", conf.jitter);
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

            update_best(0.3f, brute_side, "thompson_bandit", conf.brute);
            info.brute_step = bandit_choice;
        }

        float blended_conf = compute_adaptive_confidence(info);

        if (info.locked_side == 0 ||
            blended_conf < 0.50f ||
            HACKS->global_vars->curtime > info.lock_time + 1.0f) {
            info.locked_side = best_side;
            info.lock_time = HACKS->global_vars->curtime;
        }

        if (best_side == info.locked_side && best_score > info.confidence.brute) {
            info.lock_time = HACKS->global_vars->curtime;
        }

        info.side = info.locked_side;
        info.mode = best_mode;
        info.resolved = best_score > 0.1f;
        info.state = info.resolved ? resolver_state::RESOLVED : resolver_state::ANALYZING;

        auto logged_yaw = math::normalize_yaw(
            cur->eye_angles.y + info.max_desync * info.side);

        g_analytics.log_tick({
            p->index(),
            logged_yaw,
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
                if (info.jitter.yaw_cache.size() > 1) {
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

        auto logged_yaw = math::normalize_yaw(fired);

        g_analytics.log_tick({
            p->index(),
            logged_yaw,
            compute_adaptive_confidence(info),
            info.side,
            hit,
            HACKS->global_vars->curtime
            });

        auto features = extract_features(p, info, nullptr);
        info.ttt_net->update(features, info.side, hit);

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
        if (mode.find("jitter") != std::string::npos || mode.find("stft") != std::string::npos ||
            mode.find("ttt") != std::string::npos || mode.find("ukf") != std::string::npos ||
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
        else if (mode.find("brute") != std::string::npos || mode.find("bandit") != std::string::npos ||
            mode.find("thompson") != std::string::npos) {
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
