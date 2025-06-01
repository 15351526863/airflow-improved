#include "globals.hpp"
#include "game_movement.hpp"
#include "engine_prediction.hpp"
#include <algorithm>

namespace game_movement
{
       // yes it's pasteed..
       INLINE void accelerate(c_user_cmd& user_cmd, const vec3_t& wishdir, float wishspeed, vec3_t& velocity, float accel)
       {
               auto stored_accel = accel;
               auto currentspeed = velocity.dot(wishdir);
               const float addspeed = wishspeed - currentspeed;

               if (addspeed <= 0.f)
                       return;

               if (currentspeed < 0.f)
                       currentspeed = 0.f;

               const bool is_ducking = user_cmd.buttons.has(IN_DUCK) || HACKS->local->ducking() || HACKS->local->flags().has(FL_DUCKING);
               const bool is_walking = user_cmd.buttons.has(IN_SPEED) && !is_ducking;

               constexpr float max_speed_const = 250.f;
               auto acceleration_scale = std::max(max_speed_const, wishspeed);
               auto goal_speed = acceleration_scale;
               bool is_slow_sniper_scoped = false;

               if (HACKS->weapon && HACKS->convars.sv_accelerate_use_weapon_speed->get_int())
               {
                       const auto item_index = static_cast<std::uint16_t>(HACKS->weapon->item_definition_index());
                       is_slow_sniper_scoped = HACKS->weapon->zoom_level() > 0 &&
                                               (item_index == 11 || item_index == 38 || item_index == 9 ||
                                                item_index == 8 || item_index == 39 || item_index == 40) &&
                                               (HACKS->weapon_info->max_speed * 0.52f) < 110.f;

                       goal_speed *= std::min(1.f, (HACKS->weapon_info->max_speed / max_speed_const));

                       if ((!is_ducking && !is_walking) || ((is_walking || is_ducking) && is_slow_sniper_scoped))
                               acceleration_scale *= std::min(1.f, (HACKS->weapon_info->max_speed / max_speed_const));
               }

               if (is_ducking)
               {
                       if (!is_slow_sniper_scoped)
                               acceleration_scale *= 0.34f;

                       goal_speed *= 0.34f;
               }

               if (is_walking)
               {
                       if (!is_slow_sniper_scoped)
                               acceleration_scale *= 0.52f;

                       goal_speed *= 0.52f;
               }

               if (is_walking && currentspeed > (goal_speed - 5.f))
                       stored_accel *= std::clamp(
                               1.f - (std::max(0.f, currentspeed - (goal_speed - 5.f)) /
                                       std::max(0.f, goal_speed - (goal_speed - 5.f))), 0.f, 1.f);

               auto accelspeed =
                       stored_accel * HACKS->global_vars->interval_per_tick * acceleration_scale * HACKS->local->surface_friction();

               if (accelspeed > addspeed)
                       accelspeed = addspeed;

               velocity += wishdir * accelspeed;
       }

	INLINE void walk_move(c_user_cmd& user_cmd, vec3_t& move, vec3_t& fwd, vec3_t& right, vec3_t& velocity)
	{
		static auto sv_accelerate = HACKS->convars.sv_accelerate;

		if (fwd.z != 0.f)
			fwd = fwd.normalized();

		if (right.z != 0.f)
			right = right.normalized();

		auto max_speed = HACKS->local->is_scoped() ? HACKS->weapon_info->max_speed_alt : HACKS->weapon_info->max_speed;

		vec3_t wishvel{ fwd.x * move.x + right.x * move.y,fwd.y * move.x + right.y * move.y, 0.f };

		auto wishdir = wishvel;
		auto wishspeed = wishdir.normalized_float();
		if (wishspeed && wishspeed > max_speed)
		{
			wishvel *= max_speed / wishspeed;
			wishspeed = max_speed;
		}

		velocity.z = 0.f;
		accelerate(user_cmd, wishdir, wishspeed, velocity, sv_accelerate->get_float());
		velocity.z = 0.f;

		const auto speed_sqr = velocity.length_sqr();
		if (speed_sqr > (max_speed * max_speed))
			velocity *= max_speed / std::sqrt(speed_sqr);

		if (velocity.length() < 1.f)
			velocity = {};
	}

	INLINE void friction(vec3_t& velocity)
	{
		static auto sv_friction = HACKS->convars.sv_friction;
		static auto sv_stopspeed = HACKS->convars.sv_stopspeed;

		float speed = velocity.length_2d();
		if (speed >= 1.f)
		{
			float friction = sv_friction->get_float();
			float stop_speed = std::max< float >(speed, sv_stopspeed->get_float());
			float time = std::max< float >(HACKS->global_vars->interval_per_tick, HACKS->global_vars->frametime);
			velocity *= std::max< float >(0.f, speed - friction * stop_speed * time / speed);
		}
	}

	INLINE void full_walk_move(c_user_cmd& user_cmd, vec3_t& move, vec3_t& fwd, vec3_t& right, vec3_t& velocity)
	{
		auto unpredicted_vars = ENGINE_PREDICTION->get_unpredicted_vars();
		if (!unpredicted_vars)
			return;

		static auto sv_maxvelocity = HACKS->convars.sv_maxvelocity;

		if (unpredicted_vars->ground_entity != INT_MAX)
		{
			velocity.z = 0.f;

			friction(velocity);
			walk_move(user_cmd, move, fwd, right, velocity);

			velocity.z = 0.f;
		}

		auto max_velocity = sv_maxvelocity->get_float();
		for (int i = 0; i < 3; ++i)
		{
			auto& element = velocity[i];
			if (element > max_velocity)
				element = max_velocity;
			else if (element < -max_velocity)
				element = -max_velocity;
		}
	}

	INLINE void modify_move(c_user_cmd& user_cmd, vec3_t& velocity, float max_speed)
	{
		vec3_t viewangles;
		HACKS->engine->get_view_angles(viewangles);

		vec3_t fwd{}, right{};
		math::angle_vectors(viewangles, &fwd, &right, nullptr);

		auto cmd_movement = vec3_t{ user_cmd.forwardmove, user_cmd.sidemove, user_cmd.upmove };

		const auto speed_sqr = cmd_movement.length_sqr();
		if (speed_sqr > (max_speed * max_speed))
			cmd_movement *= max_speed / std::sqrt(speed_sqr);

		full_walk_move(user_cmd, cmd_movement, fwd, right, velocity);

		user_cmd.forwardmove = cmd_movement.x;
		user_cmd.sidemove = cmd_movement.y;
		user_cmd.upmove = cmd_movement.z;
	}

	INLINE void extrapolate(c_cs_player* player, vec3_t& origin, vec3_t& velocity, memory::bits_t& flags, bool on_ground)
	{
		static auto sv_gravity = HACKS->convars.sv_gravity;
		static auto sv_jump_impulse = HACKS->convars.sv_jump_impulse;

		if (!(flags.has(FL_ONGROUND)))
			velocity.z -= TICKS_TO_TIME(sv_gravity->get_float());
		else if (player->flags().has(FL_ONGROUND) && !on_ground)
			velocity.z = sv_jump_impulse->get_float();

		const auto src = origin;
		auto end = src + velocity * HACKS->global_vars->interval_per_tick;

		c_game_trace t;
		c_trace_filter filter;
		filter.skip = player;

		HACKS->engine_trace->trace_ray(ray_t(src, end, player->bb_mins(), player->bb_maxs()), MASK_PLAYERSOLID, &filter, &t);

		if (t.fraction != 1.f)
		{
			for (auto i = 0; i < 2; i++)
			{
				velocity -= t.plane.normal * velocity.dot(t.plane.normal);

				const auto dot = velocity.dot(t.plane.normal);
				if (dot < 0.f)
					velocity -= vec3_t{ dot* t.plane.normal.x, dot* t.plane.normal.y, dot* t.plane.normal.z };

				end = t.end + velocity * TICKS_TO_TIME(1.f - t.fraction);

				HACKS->engine_trace->trace_ray(ray_t(t.end, end, player->bb_mins(), player->bb_maxs()), MASK_PLAYERSOLID, &filter, &t);

				if (t.fraction == 1.f)
					break;
			}
		}

		origin = end = t.end;
		end.z -= 2.f;

		HACKS->engine_trace->trace_ray(ray_t(origin, end, player->bb_mins(), player->bb_maxs()), MASK_PLAYERSOLID, &filter, &t);

		flags.remove(FL_ONGROUND);

		if (t.did_hit() && t.plane.normal.z > .7f)
			flags.force(FL_ONGROUND);
	}

	INLINE memory::bits_t get_fake_jump_buttons()
	{
		memory::bits_t bits = HACKS->cmd->buttons;

		static bool last_jump = false;
		static bool fake_jump = false;

		if (!last_jump && fake_jump)
		{
			fake_jump = false;
			bits.force(IN_JUMP);
		}
		else if (bits.has(IN_JUMP))
		{
			if (HACKS->local->flags().has(FL_ONGROUND))
				fake_jump = last_jump = true;
			else
			{
				bits.remove(IN_JUMP);
				last_jump = false;
			}
		}
		else
			fake_jump = last_jump = false;

		return bits;
	}

	INLINE void force_stop()
	{
		auto velocity = HACKS->local->velocity();
		velocity.z = 0.f;

		vec3_t view_angles{};
		HACKS->engine->get_view_angles(view_angles);

		vec3_t angle;
		math::vector_angles(velocity, angle);

		float stop_speed = velocity.length();

		angle.y = view_angles.y - angle.y;

		vec3_t direction{};
		math::angle_vectors(angle, direction);

		vec3_t stop = direction * -stop_speed;

#ifdef LEGACY
		if (stop_speed > 13.f)
		{
			HACKS->cmd->forwardmove = stop.x;
			HACKS->cmd->sidemove = stop.y;
		}
		else
		{
			HACKS->cmd->forwardmove = 0.f;
			HACKS->cmd->sidemove = 0.f;
		}

#else
		HACKS->cmd->forwardmove = stop.x;
		HACKS->cmd->sidemove = stop.y;
#endif
	}

	INLINE unsigned int physics_solid_mask_for_entity(c_cs_player* entity)
	{
		static auto mp_solid_teammates = HACKS->convars.mp_solid_teammates;

		if (entity)
		{
			if (entity->life_state() == 1)
				return MASK_PLAYERSOLID_BRUSHONLY;

			if (mp_solid_teammates && mp_solid_teammates->get_int() != 2)
				return MASK_PLAYERSOLID;

			switch (entity->team())
			{
			case 2:
				return MASK_PLAYERSOLID | CONTENTS_TEAM1;
			case 3:
				return MASK_PLAYERSOLID | CONTENTS_TEAM2;
			}

			return MASK_PLAYERSOLID;
		}

		return MASK_PLAYERSOLID;
	};
}