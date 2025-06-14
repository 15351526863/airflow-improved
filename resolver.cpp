#include "globals.hpp"
#include "resolver.hpp"
#include "animations.hpp"
#include "server_bones.hpp"
#include "ragebot.hpp"
#include <array>
#include <cfloat>
#include <cstdlib>
#include <ctime>

namespace
{
	constexpr std::array<int, 3> kSideMap{ -1, 0, 1 };

	struct ArmData
	{
		int   pulls = 0;   
		float value = 0.f; 
	};

	struct ContextualBandit
	{
		static constexpr float kEpsilon = 0.10f;   
		static constexpr int   kContexts = 4;        

		std::array<std::array<ArmData, 3>, kContexts> ctx{}; // θ̂

		int select(int c)
		{
			if (static_cast<float>(std::rand()) / RAND_MAX < kEpsilon)
				return std::rand() % 3; 
			auto& arms = ctx[c];
			int    best = 0;
			float  best_val = -FLT_MAX;

			for (int a = 0; a < 3; ++a)
				if (arms[a].value > best_val)
					best_val = arms[a].value, best = a;

			return best;                
		}

		void update(int c, int a, float r)
		{
			auto& arm = ctx[c][a];
			++arm.pulls;
			arm.value += (r - arm.value) / arm.pulls;
		}
	};

	ContextualBandit g_bandit;
	bool g_seeded = ([]() { std::srand(static_cast<unsigned>(std::time(nullptr))); return true; })();
}

namespace resolver
{

	void pitch_resolve(c_cs_player* player, anim_record_t* record)
	{
		const int idx = player->index();
		auto& info = resolver_info[idx];

		if (record && record->shooting)
			return;

		const bool can_fake = (record->choke > 0) || !player->flags().has(FL_ONGROUND);

		if ((info.pitch_cycle % 2) && can_fake)
			record->eye_angles.x = -record->eye_angles.x;

		++info.pitch_cycle;
	}

	inline void detect_jitter(c_cs_player* player, resolver_info_t& resolver_info, anim_record_t* current)
	{
		auto& jitter = resolver_info.jitter;
		jitter.yaw_cache[jitter.yaw_cache_offset % YAW_CACHE_SIZE] = current->eye_angles.y;

		if (jitter.yaw_cache_offset >= YAW_CACHE_SIZE + 1)
			jitter.yaw_cache_offset = 0;
		else
			jitter.yaw_cache_offset++;

		for (int i = 0; i < YAW_CACHE_SIZE - 1; ++i)
		{
			float diff = std::fabsf(jitter.yaw_cache[i] - jitter.yaw_cache[i + 1]);
			if (diff <= 0.f)
			{
				if (jitter.static_ticks < 3)
					jitter.static_ticks++;
				else
					jitter.jitter_ticks = 0;
			}
			else if (diff >= 10.f)
			{
				if (jitter.jitter_ticks < 3)
					jitter.jitter_ticks++;
				else
					jitter.static_ticks = 0;
			}
		}

		jitter.is_jitter = jitter.jitter_ticks > jitter.static_ticks;
	}


	void resolver::jitter_resolve(c_cs_player* player, anim_record_t* current)
	{
		if (!player || !current)
			return;

		const int idx = player->index();
		auto& info = resolver_info[idx];
		auto& jitter = info.jitter;

		jitter.yaw_cache[jitter.yaw_cache_offset % YAW_CACHE_SIZE] = current->eye_angles.y;
		if (++jitter.yaw_cache_offset >= YAW_CACHE_SIZE)
			jitter.yaw_cache_offset = 0;

		const int last = (jitter.yaw_cache_offset + YAW_CACHE_SIZE - 1) % YAW_CACHE_SIZE;
		const int prev = (last + YAW_CACHE_SIZE - 1) % YAW_CACHE_SIZE;
		const float delta_now = std::fabsf(math::angle_diff(jitter.yaw_cache[last], jitter.yaw_cache[prev]));

		jitter.delta_cache[jitter.cache_offset % CACHE_SIZE] = delta_now;
		if (++jitter.cache_offset >= CACHE_SIZE)
			jitter.cache_offset = 0;

		static float ema_delta[65]{};
		static int last_big_delta_tick[65]{};

		const float alpha = 0.6f;
		ema_delta[idx] = ema_delta[idx] == 0.f ? delta_now : alpha * delta_now + (1.f - alpha) * ema_delta[idx];

		if (delta_now >= 7.f)
			last_big_delta_tick[idx] = 0;
		else
			++last_big_delta_tick[idx];

		float avg_delta = 0.f;
		float var_delta = 0.f;
		int filled = 0;

		for (int i = 0; i < CACHE_SIZE; ++i)
		{
			const float d = jitter.delta_cache[i];
			if (d != 0.f)
			{
				avg_delta += d;
				++filled;
			}
		}

		avg_delta = filled ? avg_delta / static_cast<float>(filled) : delta_now;

		for (int i = 0; i < CACHE_SIZE; ++i)
		{
			const float d = jitter.delta_cache[i];
			if (d != 0.f)
				var_delta += (d - avg_delta) * (d - avg_delta);
		}

		var_delta = filled ? var_delta / static_cast<float>(filled) : 0.f;

		float delta_for_sim = ema_delta[idx];
		if (var_delta > 25.f)
			delta_for_sim = avg_delta;

		if (last_big_delta_tick[idx] > 2)
			delta_for_sim *= 0.5f;

		if (delta_for_sim < 1.f)
			delta_for_sim = 0.f;

		auto simulate = [&](float start_yaw, float delta, int steps) -> float
			{
				float yaw = start_yaw;
				int sign = 1;
				float max_diff = 0.f;
				for (int i = 0; i < steps; ++i)
				{
					yaw += delta * sign;
					const float diff = std::fabsf(math::angle_diff(start_yaw, yaw));
					if (diff > max_diff)
						max_diff = diff;
					sign = -sign;
				}
				return max_diff;
			};

		int step_count = var_delta > 20.f ? 5 : (last_big_delta_tick[idx] > 3 ? 2 : 3);
		current->jitter_diff = simulate(current->eye_angles.y, delta_for_sim, step_count);
	}


	int resolver::brute_force(c_cs_player* player, resolver_info_t& info, int misses)
	{
		const int ctx =
			(info.jitter.is_jitter ? 1 : 0)
			| (info.is_legit() ? 2 : 0);

		const int arm = g_bandit.select(ctx);
		const int side = kSideMap[arm];

		static int last_misses[65] = {};
		const int  idx = player->index();
		const int  delta = misses - last_misses[idx];
		last_misses[idx] = misses;

		const float reward = (delta > 0) ? 0.f : 1.f; 

		g_bandit.update(ctx, arm, reward);

		info.side = side;
		return side;
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

		if (current->choke < 2)
			info.add_legit_ticks();
		else
			info.add_fake_ticks();

		if (info.is_legit())
		{
			info.resolved = false;
			info.mode = XOR("no fake");
			return;
		}

		detect_jitter(player, info, current);

		auto& jitter = info.jitter;

		if (jitter.is_jitter)
		{
			auto& misses = RAGEBOT->missed_shots[player->index()];
			if (misses > 0)
				info.side = 1337;
			else
			{
				jitter_resolve(player, current);
			}

			info.resolved = true;
			info.mode = XOR("jitter");
		}
		else
		{
			auto& misses = RAGEBOT->missed_shots[player->index()];
			if (misses > 0)
			{
				info.side = brute_force(player, info, misses);
				info.resolved = true;
				info.mode = XOR("brute");        
			}
			else
			{
				info.side = 0;
				info.mode = XOR("static");

				info.resolved = true;
			}
		}
	}

	inline void apply(c_cs_player* player, anim_record_t* current, int choke)
	{
		auto& info = resolver_info[player->index()];
		if (!HACKS->weapon_info || !HACKS->local || !HACKS->local->is_alive() || !info.resolved || info.side == 1337 || player->is_teammate(false))
			return;

		auto state = player->animstate();
		if (!state)
			return;

		float desync_angle = choke < 2 ? state->get_max_rotation() : 120.f;
		state->abs_yaw = math::normalize_yaw(player->eye_angles().y + desync_angle * info.side);

		if (RAGEBOT->missed_shots[player->index()] == 0)
			resolver::pitch_resolve(player, current);
	}
}
