#include "globals.hpp"
#include "resolver.hpp"
#include "animations.hpp"
#include "server_bones.hpp"
#include "ragebot.hpp"
#include "penetration.hpp"
#include "LSTM.hpp"
#include <array>
#include <filesystem>
#include <cfloat>
#include <cstdlib>
#include <ctime>
#include <cmath>

namespace
{
	constexpr std::array<int, 3> kSideMap{ -1,0,1 };

	struct ArmData
	{
		int   pulls = 0;
		float value = 0.f;
	};

	struct ContextualBandit
	{
		static constexpr int kContexts = 8;
		std::array<std::array<ArmData, 3>, kContexts> ctx{};

		int select(int c)
		{
			auto& arms = ctx[c];
			for (int a = 0; a < 3; ++a)
				if (arms[a].pulls == 0)
					return a;
			int   total = arms[0].pulls + arms[1].pulls + arms[2].pulls;
			float logt = std::log(static_cast<float>(total));
			int   best = 0;
			float best_ucb = -FLT_MAX;
			for (int a = 0; a < 3; ++a)
			{
				float ucb = arms[a].value + std::sqrt(2.f * logt / arms[a].pulls);
				if (ucb > best_ucb)
				{
					best_ucb = ucb;
					best = a;
				}
			}
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
        LSTM g_jitter_lstm{};
        bool g_seeded = ([]() {
                std::srand(static_cast<unsigned>(std::time(nullptr)));
                return true;
                })();
        bool g_lstm_loaded = ([]() {
                resolver::load_lstm_weights();
                return true;
                })();
}

namespace resolver
{
        void save_lstm_weights()
        {
                std::filesystem::create_directories("airflow/models");
                g_jitter_lstm.save("airflow/models/jitter_lstm.bin");
        }

        void load_lstm_weights()
        {
                std::filesystem::create_directories("airflow/models");
                g_jitter_lstm.load("airflow/models/jitter_lstm.bin");
        }

        void train_lstm(const std::vector<double>& input, double target)
        {
                g_jitter_lstm.train(input, target);
        }
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


        inline int jitter_fix(c_cs_player* player, anim_record_t* current)
        {
                if (!player || !current)
                        return 0;

                const int idx = player->index();
                auto& info = resolver_info[idx];
                auto      anim = ANIMFIX->get_anims(idx);

                if (!anim || anim->records.size() < 2)
                        return 0;

                auto& prev = anim->records[1];
                auto& jit = info.jitter;

                const double eye_delta = math::normalize_yaw(current->eye_angles.y - prev.eye_angles.y);
                const double abs_delta = std::fabs(eye_delta);

                jit.delta_cache[jit.cache_offset % CACHE_SIZE] = abs_delta;
                ++jit.cache_offset;

                std::vector<double> input{ eye_delta, abs_delta };
                jit.last_input[0] = eye_delta;
                jit.last_input[1] = abs_delta;

                double out = g_jitter_lstm.forward(input);
                int side = out >= 0.5 ? 1 : -1;

                info.side = side;
                return side;
        }

	inline int brute_force(c_cs_player* player, resolver_info_t& info, int misses)
	{
		bool crit = player->animlayers()[ANIMATION_LAYER_MOVEMENT_LAND_OR_CLIMB].weight > 0.f;
		int ctx = (info.jitter.is_jitter ? 1 : 0)
			| (info.is_legit() ? 2 : 0)
			| (crit ? 4 : 0);

		int arm = g_bandit.select(ctx);
		int side = kSideMap[arm];

		static int last_misses[65]{};
		int idx = player->index();
		int delta = misses - last_misses[idx];
		last_misses[idx] = misses;

		float reward = delta > 0 ? 0.f : 1.f;
		g_bandit.update(ctx, arm, reward);

		info.side = side;
		return side;
	}


	static float static_resolve(c_cs_player* player)
	{
		auto state = player->animstate();
		if (!state)
			return 0.f;

		float max_rot = state->get_max_rotation();
		float delta = math::angle_diff(state->eye_yaw, state->abs_yaw);

		if (delta > max_rot)
			state->abs_yaw = math::normalize_yaw(state->eye_yaw - max_rot);

		if (delta < -max_rot)
			state->abs_yaw = math::normalize_yaw(state->eye_yaw + max_rot);

		if (state->on_ground)
		{
			if (state->velocity_length_xy > 0.1f)
			{
				state->abs_yaw = math::approach_angle(state->eye_yaw, state->abs_yaw,
					state->last_update_increment * (30.f + 20.f * state->walk_run_transition));

				if (state->velocity_length_xy > 250.f)
					state->abs_yaw = math::approach_angle(state->eye_yaw, state->abs_yaw,
						state->last_update_increment * state->velocity_length_xy);
			}
			else
				state->abs_yaw = math::approach_angle(player->lower_body_yaw(), state->abs_yaw,
					state->last_update_increment * 100.f);
		}
		else if (state->velocity_length_xy > 250.f)
			state->abs_yaw = math::approach_angle(state->eye_yaw, state->abs_yaw,
				state->last_update_increment * state->velocity_length_xy);

		state->abs_yaw = math::normalize_yaw(state->abs_yaw);
		return state->abs_yaw;
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
				info.side = jitter_fix(player, current);

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
				info.resolved_yaw = static_resolve(player);
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

		if (info.mode == XOR("static"))
		{
			state->abs_yaw = info.resolved_yaw;
		}
		else
		{
			if (info.side == 1337)
				return;

			float desync = choke < 2 ? state->get_max_rotation() : 120.f;
			state->abs_yaw = math::normalize_yaw(player->eye_angles().y + desync * info.side);
		}

		if (RAGEBOT->missed_shots[player->index()] == 0)
			resolver::pitch_resolve(player, current);
	}
}
