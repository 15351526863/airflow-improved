#pragma once
#include "animations.hpp"
#include <cstring>
#include <algorithm>

constexpr int CACHE_SIZE = 2;
constexpr int YAW_CACHE_SIZE = 16;
constexpr auto MAX_TICKS = 3;

struct resolver_info_t
{
    bool resolved{};
    int side{};

    int legit_ticks{};
    int fake_ticks{};

    INLINE void add_legit_ticks()
    {
        if (legit_ticks >= MAX_TICKS)
        {
            fake_ticks = 0;
            legit_ticks = MAX_TICKS;
        }
        else
            ++legit_ticks;
    }

    INLINE void add_fake_ticks()
    {
        if (fake_ticks >= MAX_TICKS)
        {
            legit_ticks = 0;
            fake_ticks = MAX_TICKS;
        }
        else
            ++fake_ticks;
    }

    INLINE bool is_legit()
    {
        return legit_ticks > fake_ticks;
    }

    std::string mode{};
    c_animation_layers initial_layers[13]{};

    struct jitter_info_t
    {
        bool is_jitter{};
        bool high_freq_jitter{};
        int frequency{};

        struct jitter_state_machine_t
        {
            enum state_t { STATIC, JITTER } state{ STATIC };
            int ticks_static{};
            int ticks_jitter{};

            INLINE void reset()
            {
                state = STATIC;
                ticks_static = 0;
                ticks_jitter = 0;
            }

            INLINE void update(bool is_jitter_now, float variance)
            {
                const bool spike = variance > 20.f;
                if (is_jitter_now || spike)
                {
                    ++ticks_jitter;
                    ticks_static = std::max(0, ticks_static - 1);
                }
                else
                {
                    ++ticks_static;
                    ticks_jitter = std::max(0, ticks_jitter - 1);
                }

                if (spike || ticks_jitter > 2)
                    state = JITTER;
                else if (ticks_static > 2)
                    state = STATIC;
            }

            INLINE bool is_jitter() const { return state == JITTER; }
        } machine{};

        static constexpr int DELTA_WINDOW = 16;
        float delta_history[DELTA_WINDOW]{};
        int delta_offset{};
        float variance{};
        float autocorr{};

        // FFT buffers, index 0 stores the DC component and is unused in analysis
        float fft_real[DELTA_WINDOW]{};
        float fft_imag[DELTA_WINDOW]{};

        float delta_cache[CACHE_SIZE]{};
        int cache_offset{};

        float yaw_cache[YAW_CACHE_SIZE]{};
        int yaw_cache_offset{};

        float delta_variance{};

        int jitter_ticks{};
        int static_ticks{};

        __forceinline void reset()
        {
            is_jitter = false;
            high_freq_jitter = false;
            frequency = 0;

            delta_offset = 0;
            variance = 0.f;
            autocorr = 0.f;

            cache_offset = 0;
            yaw_cache_offset = 0;

            jitter_ticks = 0;
            static_ticks = 0;

            delta_variance = 0.f;

            machine.reset();

            std::memset(delta_cache, 0, sizeof(delta_cache));
            std::memset(yaw_cache, 0, sizeof(yaw_cache));
            std::memset(delta_history, 0, sizeof(delta_history));
            std::memset(fft_real, 0, sizeof(fft_real));
            std::memset(fft_imag, 0, sizeof(fft_imag));
        }

        INLINE bool is_high_jitter() const {
            return variance > 15.f && autocorr < 0.f;
        }
    } jitter;

    struct extended_desync_estimator_t
    {
        float yaw{};
        float vel{};
        float accel{};

        float err_yaw{ 25.f };
        float err_vel{ 15.f };
        float err_accel{ 10.f };

        float proc_noise{ 1.5f };
        float meas_noise{ 10.f };

        INLINE void reset()
        {
            yaw = vel = accel = 0.f;
            err_yaw = 25.f;
            err_vel = 15.f;
            err_accel = 10.f;
        }

        INLINE void update(float delta_yaw, float speed)
        {
            yaw += vel + 0.5f * accel;
            vel += accel;
            err_yaw += proc_noise;
            err_vel += proc_noise;
            err_accel += proc_noise * 0.5f;

            float noise = meas_noise + std::fabs(speed) * 0.3f;
            float k_yaw = err_yaw / (err_yaw + noise);
            float k_vel = err_vel / (err_vel + noise * 0.5f);

            float resid = delta_yaw - yaw;
            yaw += k_yaw * resid;
            vel += k_vel * resid;
            accel = math::lerp(0.5f, accel, resid);

            err_yaw *= (1.f - k_yaw);
            err_vel *= (1.f - k_vel);
            err_accel *= 0.9f;
        }

        INLINE float estimate() const { return yaw; }
    } desync_estimator;

    struct freestanding_t
    {
        bool updated{};
        int side{};
        float update_time{};

        float left_fraction{};
        float right_fraction{};

        inline void reset()
        {
            updated = false;
            side = 0;
            update_time = 0.f;
            left_fraction = right_fraction = 0.f;
        }
    } freestanding{};

    struct side_confidence_t {
        float jitter{};
        float foot_delta{};
        float freestand{};
        float velocity{};
        float brute{};

        inline void reset() {
            jitter = foot_delta = freestand = velocity = brute = 0.f;
        }
    } confidence{};

    struct predictor_success_t {
        float jitter{ 1.f };
        float foot_delta{ 1.f };
        float freestand{ 1.f };
        float velocity{ 1.f };
        float brute{ 1.f };

        inline void clamp() {
            jitter = std::clamp(jitter, 0.1f, 2.f);
            foot_delta = std::clamp(foot_delta, 0.1f, 2.f);
            freestand = std::clamp(freestand, 0.1f, 2.f);
            velocity = std::clamp(velocity, 0.1f, 2.f);
            brute = std::clamp(brute, 0.1f, 2.f);
        }
    } accuracy{};

    struct side_probability_t
    {
        float prob_left{ 0.5f };
        float prob_right{ 0.5f };

        INLINE void normalize()
        {
            float sum = prob_left + prob_right;
            if (sum > 0.f)
            {
                prob_left /= sum;
                prob_right /= sum;
            }
        }

        INLINE void update(bool hit, int predicted_side, float velocity, float desync)
        {
            float dyn = std::clamp((velocity + std::fabs(desync)) / 150.f, 0.f, 1.f);
            float alpha = hit ? 0.7f + 0.3f * dyn : 0.3f + 0.2f * dyn;

            if (predicted_side > 0)
                prob_right = prob_right * alpha + prob_left * (1.f - alpha);
            else
                prob_left = prob_left * alpha + prob_right * (1.f - alpha);

            normalize();
        }

        INLINE int best_side() const
        {
            return prob_left > prob_right ? -1 : 1;
        }

        INLINE void reset()
        {
            prob_left = prob_right = 0.5f;
        }
    } side_prob{};

    float side_prob_last_time{};

    float max_desync{};

    int brute_step{};
    int brute_cycle{};
    float brute_offsets[7]{};
    int brute_hits[7]{};

    struct brute_pattern_t {
        int last_step{ -1 };
        int second_last_step{ -1 };

        inline void reset()
        {
            last_step = -1;
            second_last_step = -1;
        }

        inline void update(int step)
        {
            second_last_step = last_step;
            last_step = step;
        }

        inline int predict_next(int max_steps) const
        {
            if (last_step == second_last_step && last_step != -1)
                return (last_step + max_steps / 2) % max_steps;
            return (last_step + 1) % max_steps;
        }
    } brute_pattern{};

    int locked_side{};
    float lock_time{};

#ifdef LEGACY
    int lby_breaker{};
    int lby_update{};

    struct move_t
    {
        float time{};
        float lby{};

        inline void reset()
        {
            time = 0.f;
            lby = 0.f;
        }
    } move{};

    struct lby_flicks_t
    {
        bool lby_breaker_failed = false;

        float last_lby_value = 0.0f;
        float next_lby_update = 0.0f;

        int logged_lby_delta_score = 0;
        float logged_lby_delta = 0.0f;

        c_animation_layers old_layers[13]{};

        inline void reset()
        {
            lby_breaker_failed = false;
            last_lby_value = 0.f;
            next_lby_update = 0.f;

            logged_lby_delta_score = 0;
            logged_lby_delta = 0.f;

            for (auto& i : old_layers)
                i = {};
        }
    } lby{};
#endif

    anim_record_t record{};

    inline void reset()
    {
        resolved = false;
        side = 0;
        legit_ticks = 0;
        fake_ticks = 0;

        mode = "";

        freestanding.reset();
        jitter.reset();
        desync_estimator.reset();
        confidence.reset();
        accuracy = {};
        side_prob.reset();
        side_prob_last_time = 0.f;
        brute_step = 0;
        brute_cycle = 0;
        max_desync = 0.f;
        locked_side = 0;
        lock_time = 0.f;
        std::memset(brute_hits, 0, sizeof(brute_hits));
        for (auto& o : brute_offsets)
            o = 0.f;
        brute_pattern.reset();
#ifdef LEGACY
        lby_breaker = 0;
        lby_update = 0;
        lby.reset();
        move.reset();
        record.reset();
#endif

        for (auto& i : initial_layers)
            i = {};
    }
};

inline resolver_info_t resolver_info[65]{};

namespace resolver
{
    INLINE void reset()
    {
        for (auto& i : resolver_info)
            i.reset();
    }

    extern void prepare_side(c_cs_player* player, anim_record_t* current, anim_record_t* last);
    extern void apply_side(c_cs_player* player, anim_record_t* current, int choke);
    extern void prepare_side_improved(c_cs_player* player, anim_record_t* current, anim_record_t* last);
    extern void apply_side_improved(c_cs_player* player, anim_record_t* current, int choke);
    extern void brute_resolve_adaptive(c_cs_player* player, resolver_info_t& info, anim_record_t* current, float last_miss);
    extern void brute_resolve_dynamic(c_cs_player* player, resolver_info_t& info, anim_record_t* current, float last_miss_delta);
    extern void brute_resolve_gradient(c_cs_player* player, resolver_info_t& info, anim_record_t* current, float miss_delta);
    extern void on_shot_result(c_cs_player* player, bool hit, int step,
        float fired_yaw, float impact_yaw);
    extern void prepare_bones_with_resolver(c_cs_player* player, anim_record_t* record);
}