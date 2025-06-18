#pragma once
#include "animations.hpp"
#include "LSTM.hpp"

constexpr int CACHE_SIZE = 2;
constexpr int YAW_CACHE_SIZE = 8;
constexpr auto MAX_TICKS = 3;
constexpr float JITTER_BEGIN_ANGLE = 6.f;

struct resolver_info_t
{
	bool resolved{};
	float resolved_yaw{};
	int side{};
	int pitch_cycle{};

	int legit_ticks{};
	int fake_ticks{};

	INLINE void add_legit_ticks()
	{
		if (legit_ticks < MAX_TICKS)
			++legit_ticks;
		else
			fake_ticks = 0;
	}

	INLINE void add_fake_ticks()
	{
		if (fake_ticks < MAX_TICKS)
			++fake_ticks;
		else
			legit_ticks = 0;
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

		float delta_cache[CACHE_SIZE]{};
		int cache_offset{};

		float yaw_cache[YAW_CACHE_SIZE]{};
		int yaw_cache_offset{};

		int jitter_ticks{};
		int static_ticks{};

                int jitter_tick{};

                double last_input[2]{};

		__forceinline void reset()
		{
			is_jitter = false;

			cache_offset = 0;
			yaw_cache_offset = 0;

			jitter_ticks = 0;
			static_ticks = 0;

                        jitter_tick = 0;

                        std::memset(delta_cache, 0, sizeof(delta_cache));
                        std::memset(yaw_cache, 0, sizeof(yaw_cache));
                        last_input[0] = last_input[1] = 0.0;
                }
        } jitter;

	struct freestanding_t
	{
		bool updated{};
		int side{};
		float update_time{};

		inline void reset()
		{
			updated = false;
			side = 0;
			update_time = 0.f;
		}
	} freestanding{};

	anim_record_t record{};

	inline void reset()
	{
		resolved = false;
		side = 0;
		legit_ticks = 0;
		fake_ticks = 0;
		pitch_cycle = 0;
		resolved_yaw = 0.f;

		mode = "";

		freestanding.reset();
		jitter.reset();

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

	extern int jitter_fix(c_cs_player* player, anim_record_t* current);
	extern void prepare_side(c_cs_player* player, anim_record_t* current, anim_record_t* last);
        extern void apply(c_cs_player* player, anim_record_t* current, int choke);
        int brute_force(c_cs_player* player, resolver_info_t& info, int misses);
        void save_lstm_weights();
        void load_lstm_weights();
        void train_lstm(const std::vector<double>& input, double target);
}
