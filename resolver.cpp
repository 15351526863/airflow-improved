#include "globals.hpp"
#include "resolver.hpp"
#include "animations.hpp"
#include "server_bones.hpp"
#include "ragebot.hpp"

namespace resolver
{
	void pitch_resolve(c_cs_player* player, anim_record_t* record)
	{
		const int idx = player->index();
		auto& info = resolver_info[idx];

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
				float first_angle = math::normalize_yaw(jitter.yaw_cache[YAW_CACHE_SIZE - 1]);
				float second_angle = math::normalize_yaw(jitter.yaw_cache[YAW_CACHE_SIZE - 2]);

				float _first_angle = std::sin(DEG2RAD(first_angle));
				float _second_angle = std::sin(DEG2RAD(second_angle));

				float __first_angle = std::cos(DEG2RAD(first_angle));
				float __second_angle = std::cos(DEG2RAD(second_angle));

				float avg_yaw = math::normalize_yaw(RAD2DEG(std::atan2f((_first_angle + _second_angle) / 2.f, (__first_angle + __second_angle) / 2.f)));
				float diff = math::normalize_yaw(current->eye_angles.y - avg_yaw);

				info.side = diff > 0.f ? -1 : 1;
			}

			info.resolved = true;
			info.mode = XOR("jitter");
		}
		else
		{
			auto& misses = RAGEBOT->missed_shots[player->index()];
			if (misses > 0)
			{
				switch (misses % 3)
				{
				case 1:
					info.side = -1;
					break;
				case 2:
					info.side = 1;
					break;
				case 0:
					info.side = 0;
					break;
				}

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

	inline void apply_side(c_cs_player* player, anim_record_t* current, int choke)
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
