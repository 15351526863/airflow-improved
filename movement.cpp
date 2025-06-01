#include "globals.hpp"
#include "movement.hpp"
#include "animations.hpp"
#include "server_bones.hpp"
#include "ragebot.hpp"

static vec3_t ray_circle_intersection(const vec3_t& ray, const vec3_t& center, float r)
{
       if (std::fabsf(ray.x) > std::fabsf(ray.y))
       {
               float k = ray.y / ray.x;

               float a = 1.f + k * k;
               float b = -2.f * center.x - 2.f * k * center.y;
               float c = center.length_sqr_2d() - r * r;

               float d = b * b - 4.f * a * c;

               if (d < 0.f)
               {
                       vec3_t nearest_on_ray = ray * center.dot(ray);
                       vec3_t diff = (nearest_on_ray - center).normalized();
                       return center + diff * r;
               }
               else if (d < 0.001f)
               {
                       float x = -b / (2.f * a);
                       float y = k * x;
                       return { x, y, 0.f };
               }

               float d_sqrt = std::sqrtf(d);

               float x = (-b + d_sqrt) / (2.f * a);
               float y = k * x;

               vec3_t dir1(x, y, 0.f);

               x = (-b - d_sqrt) / (2.f * a);
               y = k * x;

               vec3_t dir2(x, y, 0.f);

               if (ray.dot(dir1) > ray.dot(dir2))
                       return dir1;

               return dir2;
       }
       else
       {
               float k = ray.x / ray.y;

               float a = 1.f + k * k;
               float b = -2.f * center.y - 2.f * k * center.x;
               float c = center.length_sqr_2d() - r * r;

               float d = b * b - 4.f * a * c;

               if (d < 0.f)
               {
                       vec3_t nearest_on_ray = ray * center.dot(ray);
                       vec3_t diff = (nearest_on_ray - center).normalized();
                       return center + diff * r;
               }
               else if (d < 0.001f)
               {
                       float y = -b / (2.f * a);
                       float x = k * y;
                       return { x, y, 0.f };
               }

               float d_sqrt = std::sqrtf(d);

               float y = (-b + d_sqrt) / (2.f * a);
               float x = k * y;

               vec3_t dir1(x, y, 0.f);

               y = (-b - d_sqrt) / (2.f * a);
               x = k * y;

               vec3_t dir2(x, y, 0.f);

               if (ray.dot(dir1) > ray.dot(dir2))
                       return dir1;

               return dir2;
       }
}

static float calculate_throw_yaw(const vec3_t& wish_dir, const vec3_t& vel, float throw_velocity, float throw_strength)
{
       vec3_t dir_normalized = wish_dir;
       dir_normalized.z = 0.f;
       dir_normalized = dir_normalized.normalized();

       float cos_pitch = dir_normalized.dot(wish_dir) / std::sqrtf(wish_dir.length_sqr());

       float speed = std::clamp(throw_velocity * 0.9f, 15.f, 750.f) * (std::clamp(throw_strength, 0.f, 1.f) * 0.7f + 0.3f) * cos_pitch;
       vec3_t real_dir = ray_circle_intersection(dir_normalized, vel * 1.25f, speed) - vel * 1.25f;

       vec3_t ang{};
       math::vector_angles(real_dir, ang);
       return ang.y;
}

static float calculate_throw_pitch(const vec3_t& wish_dir, float wish_z_vel, const vec3_t& vel, float throw_velocity, float throw_strength)
{
       float speed = std::clamp(throw_velocity * 0.9f, 15.f, 750.f) * (std::clamp(throw_strength, 0.f, 1.f) * 0.7f + 0.3f);

       vec3_t cur_vel = vel * 1.25f + wish_dir * speed;
       vec3_t wish_vel = vec3_t(vel.x, vel.y, wish_z_vel) * 1.25f + wish_dir * speed;

       vec3_t ang1{}, ang2{};
       math::vector_angles(cur_vel, ang1);
       math::vector_angles(wish_vel, ang2);

       float ang_diff = ang2.x - ang1.x;

       return ang_diff * (std::cos(DEG2RAD(ang_diff)) + 1.f) * 0.5f;
}

void c_movement::update_ground_ticks()
{
	if (HACKS->local->flags().has(FL_ONGROUND))
		ground_ticks++;
	else
		ground_ticks = 0;
}

void c_movement::auto_jump()
{
	if (HACKS->local->move_type() == MOVETYPE_LADDER || HACKS->local->move_type() == MOVETYPE_NOCLIP)
		return;

	if (!g_cfg.misc.auto_jump)
		return;

    if (!last_jumped && should_fake)
    {
        should_fake = false;
        HACKS->cmd->buttons.force(IN_JUMP);
    }
    else if (HACKS->cmd->buttons.has(IN_JUMP))
    {
        if (HACKS->local->flags().has(FL_ONGROUND))
            should_fake = last_jumped = true;
        else
        {
            HACKS->cmd->buttons.remove(IN_JUMP);
            last_jumped = false;
        }
    }
    else
    {
        should_fake = last_jumped = false;
    }
}

void c_movement::auto_strafe()
{
	if (!g_cfg.misc.auto_strafe)
		return;

	if (RAGEBOT->trigger_stop && (RAGEBOT->rage_config.quick_stop_options & in_air))
		return;

	if (HACKS->local->move_type() == MOVETYPE_LADDER || HACKS->local->move_type() == MOVETYPE_NOCLIP)
		return;

	if (HACKS->local->flags().has(FL_ONGROUND))
		return;

	if (HACKS->cmd->buttons.has(IN_SPEED))
		return;

	auto holding_w = HACKS->cmd->buttons.has(IN_FORWARD);
	auto holding_a = HACKS->cmd->buttons.has(IN_MOVELEFT);
	auto holding_s = HACKS->cmd->buttons.has(IN_BACK);
	auto holding_d = HACKS->cmd->buttons.has(IN_MOVERIGHT);

	auto pressing_move = holding_w || holding_a || holding_s || holding_d;

	if (!pressing_move)
		return;

	vec3_t velocity = HACKS->local->velocity();
	velocity.z = 0.f;

	auto speed = velocity.length();
	auto ideal_strafe = (speed > 5.f) ? RAD2DEG(std::asin(15.f / speed)) : 90.f;
	ideal_strafe *= 1.f - (g_cfg.misc.strafe_smooth * 0.01f);

	ideal_strafe = std::min(90.f, ideal_strafe);

	switch_key *= -1.f;

	float wish_dir{};
	if (pressing_move)
	{
		if (holding_w)
		{
			if (holding_a)
				wish_dir += (STRAFE_LEFT / 2);
			else if (holding_d)
				wish_dir += (STRAFE_RIGHT / 2);
			else
				wish_dir += STRAFE_FORWARDS;
		}
		else if (holding_s)
		{
			if (holding_a)
				wish_dir += STRAFE_BACK_LEFT;
			else if (holding_d)
				wish_dir += STRAFE_BACK_RIGHT;
			else
				wish_dir += STRAFE_BACKWARDS;

			HACKS->cmd->forwardmove = 0.f;
		}
		else if (holding_a)
			wish_dir += STRAFE_LEFT;
		else if (holding_d)
			wish_dir += STRAFE_RIGHT;

		base_view_angle.y += math::normalize_yaw(wish_dir);
	}

	auto smooth = (1.f - (0.15f * (g_cfg.misc.strafe_smooth * 0.01f)));
	auto forward_speed = HACKS->convars.cl_forwardspeed->get_float();
	auto side_speed = HACKS->convars.cl_sidespeed->get_float();

	if (speed <= 0.5f)
	{
		HACKS->cmd->forwardmove = forward_speed;
		return;
	}

	const auto diff = math::normalize_yaw(base_view_angle.y - RAD2DEG(std::atan2f(velocity.y, velocity.x)));

	HACKS->cmd->forwardmove = std::clamp((5850.f / speed), -forward_speed, forward_speed);
	HACKS->cmd->sidemove = (diff > 0.f) ? -side_speed : side_speed;

	base_view_angle.y = math::normalize_yaw(base_view_angle.y - diff * smooth);
}

void c_movement::fast_stop()
{
	/*if (RAGEBOT->working || RAGEBOT->best_rage_player.valid)
	{
		complete_fast_stop = false;
		return;
	}*/

	if (!g_cfg.misc.fast_stop && !g_cfg.binds[ap_b].toggled) 
	{
		complete_fast_stop = false;
		return;
	}

	if (!on_ground()) 
	{
		complete_fast_stop = false;
		return;
	}

	vec3_t velocity = HACKS->local->velocity();
	if (HACKS->cmd->buttons.has(IN_MOVELEFT)
		|| HACKS->cmd->buttons.has(IN_MOVERIGHT)
		|| HACKS->cmd->buttons.has(IN_BACK)
		|| HACKS->cmd->buttons.has(IN_FORWARD)) 
	{
		complete_fast_stop = false;
		return;
	}

	if (complete_fast_stop)
		return;

	instant_stop();

	if (velocity.length_2d() <= 15.f)
		complete_fast_stop = true;
}

void c_movement::edge_jump( )
{
	
}

bool c_movement::can_use_auto_peek()
{
	if (!g_cfg.binds[ap_b].toggled)
		return false;

	auto moving = HACKS->cmd->buttons.has(IN_MOVELEFT)
		|| HACKS->cmd->buttons.has(IN_MOVERIGHT)
		|| HACKS->cmd->buttons.has(IN_BACK)
		|| HACKS->cmd->buttons.has(IN_FORWARD);

	auto origin = HACKS->local->get_abs_origin();

	auto& info = peek_info;
	if ( !info.start_pos.valid( ) ) {
		if ( !( HACKS->local->flags( ).has( FL_ONGROUND ) ) )
		{
			c_game_trace trace {};
			c_trace_filter_world_and_props_only filter {};
			HACKS->engine_trace->trace_ray( ray_t { origin, origin - vec3_t( 0.f, 0.f, 8192.f ) }, MASK_SOLID, &filter, &trace );

			info.start_pos = trace.end;
		}
		else
			info.start_pos = origin;
	}
	else
	{
		info.peek_init = true;

		auto cant_stop = RAGEBOT->working || RAGEBOT->best_rage_player.valid;

		if (on_ground())
		{
			auto misc_weapon = HACKS->weapon->is_misc_weapon()
				&& !HACKS->weapon->is_knife()
				&& !HACKS->weapon->is_taser();

			bool cmd_attack = HACKS->cmd->buttons.has(IN_ATTACK);
			bool is_firing = HACKS->weapon->item_definition_index() == WEAPON_REVOLVER ? cmd_attack && RAGEBOT->revolver_fire : cmd_attack;

			if (!misc_weapon && (is_firing || RAGEBOT->firing || (g_cfg.misc.retrack_peek && !moving)))
				info.peek_execute = true;

			if (g_cfg.misc.retrack_peek && moving && !info.old_move)
				info.peek_execute = false;
		}

		if (info.peek_execute)
		{
			auto origin_delta = info.start_pos - origin;
			auto distance = origin_delta.length_2d();

			auto return_position = math::calc_angle(origin, info.start_pos);

			if (distance > 10.f)
			{
				base_view_angle.y = math::normalize_yaw(return_position.y);

				HACKS->cmd->forwardmove = HACKS->convars.cl_forwardspeed->get_float();
				HACKS->cmd->sidemove = 0.f;
			}
			else
			{
				if (!cant_stop && HACKS->local->velocity().length_2d() > 15)
					instant_stop();
				else
					info.peek_execute = false;
			}
		}
	}

	info.old_move = moving;
	return true;
}

void c_movement::auto_peek()
{
	if (!can_use_auto_peek())
	{
		peek_info.reset();
		return;
	}
}

void c_movement::super_toss()
{
       if (!g_cfg.misc.compensate_throwable)
               return;

       if (!HACKS->weapon || !HACKS->weapon_info || !HACKS->weapon->is_grenade())
               return;

       vec3_t direction{};
       math::angle_vectors(HACKS->cmd->viewangles, direction);

       vec3_t smoothed_velocity = (local_velocity + last_local_velocity) * 0.5f;

       float base_speed = std::clamp(HACKS->weapon_info->throw_velocity * 0.9f, 15.f, 750.f) *
               (std::clamp(HACKS->weapon->throw_strength(), 0.f, 1.f) * 0.7f + 0.3f);

       vec3_t base_vel = direction * base_speed;
       vec3_t current_vel = local_velocity * 1.25f + base_vel;

       vec3_t target_vel = (base_vel + smoothed_velocity * 1.25f).normalized();
       if (current_vel.dot(direction) > 0.f)
               target_vel = direction;

       float yaw = calculate_throw_yaw(target_vel, local_velocity, HACKS->weapon_info->throw_velocity, HACKS->weapon->throw_strength());

       if (!HACKS->local->flags().has(FL_ONGROUND))
               HACKS->cmd->viewangles.y = yaw;

       HACKS->cmd->viewangles.x += calculate_throw_pitch(direction, std::clamp(local_velocity.z, -120.f, 120.f), local_velocity,
               HACKS->weapon_info->throw_velocity, HACKS->weapon->throw_strength());
}

void c_movement::instant_stop()
{
	vec3_t view_angle{};
	HACKS->engine->get_view_angles(view_angle);

	vec3_t angle{};
	math::vector_angles(HACKS->local->velocity(), angle);

	float speed = HACKS->local->velocity().length();
	angle.y = view_angle.y - angle.y;

	vec3_t direction;
	math::angle_vectors(angle, direction);

	vec3_t stop = direction * -speed;

	HACKS->cmd->forwardmove = stop.x;
	HACKS->cmd->sidemove = stop.y;
}

void c_movement::rotate_movement(c_user_cmd* cmd, vec3_t& ang)
{
	if (HACKS->local->move_type() == MOVETYPE_LADDER || HACKS->local->move_type() == MOVETYPE_NOCLIP)
		return;

	vec3_t direction{};
	vec3_t move_angle{};

	auto move = vec3_t{ cmd->forwardmove, cmd->sidemove, 0 };
	auto length = move.normalized_float();
	if (length == 0.f)
		return;

	math::vector_angles(move, move_angle);

	auto delta = (cmd->viewangles.y - ang.y);

	move_angle.y += delta;

	math::angle_vectors(move_angle, direction);

	direction *= length;

	if (cmd->viewangles.x < -90 || cmd->viewangles.x > 90)
		direction.x = -direction.x;

	cmd->forwardmove = direction.x;
	cmd->sidemove = direction.y;

	float negative = calc_move_angle(cmd, 180.f - cmd->viewangles.z);
	float positive = calc_move_angle(cmd, cmd->viewangles.z + 180.f);

	float step = cmd->viewangles.x / 89.f;

	if (std::fabsf(cmd->viewangles.z) > 0.f)
		cmd->forwardmove = std::lerp(cmd->forwardmove, step <= 0.f ? positive : negative, step);

	if (!g_cfg.misc.slide_walk)
		cmd->buttons.remove(IN_FORWARD | IN_BACK | IN_MOVERIGHT | IN_MOVELEFT);
	else
	{
		if (cmd->sidemove < 5.f)
			cmd->buttons.force(IN_MOVERIGHT);
		else if (cmd->sidemove > -5.f)
			cmd->buttons.force(IN_MOVELEFT);

		if (cmd->forwardmove < 5.f)
			cmd->buttons.force(IN_FORWARD);
		else if (cmd->forwardmove > -5.f)
			cmd->buttons.force(IN_BACK);
	}
}

void c_movement::run()
{
	HACKS->engine->get_view_angles(base_view_angle);

	update_ground_ticks();
	auto_jump();
}

void c_movement::run_predicted()
{
        if (!HACKS->weapon)
                return;

        edge_jump();
        auto_strafe();
        fast_stop();
        auto_peek();
        super_toss();
}

void c_movement::render_peek_position()
{
	if (!HACKS->local || !HACKS->local->is_alive())
		return;

	auto& info = peek_info;
	if (!info.valid())
		return;

	info.render_points.clear();
	info.render_points.reserve(100);

	constexpr float step = M_PI * 2.0f / 60;

	vec2_t center_pos{};

	if (RENDER->world_to_screen(info.start_pos, center_pos, true))
		info.render_points.emplace_back(ImVec2(center_pos.x, center_pos.y));

	for (float lat = 0.f; lat <= M_PI * 2.0f; lat += step)
	{
		const auto& point3d = vec3_t(std::sin(lat), std::cos(lat), 0.f) * 30.f;
		vec2_t point2d;
		if (RENDER->world_to_screen(info.start_pos + point3d, point2d, true))
			info.render_points.emplace_back(ImVec2(point2d.x, point2d.y));
	}

	if (info.render_points.empty())
		return;

	auto center = info.render_points[0];

	auto draw_list = RENDER->get_draw_list();
	RESTORE(draw_list->Flags);

	draw_list->Flags |= ImDrawListFlags_AntiAliasedFill;

	auto first_clr = g_cfg.misc.autopeek_clr.base();
	auto second_clr = g_cfg.misc.autopeek_clr_back.base();
	auto clr = peek_info.peek_execute ? second_clr : first_clr;

        for (int i = 0; i < info.render_points.size() - 1; ++i) {
                RENDER->triangle_filled_gradient(
                        info.render_points[i].x, info.render_points[i].y,
                        info.render_points[i + 1].x, info.render_points[i + 1].y,
                        center.x, center.y,
                        clr.new_alpha(0), clr.new_alpha(0), clr);
        }
}
