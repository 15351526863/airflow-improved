﻿#include "globals.hpp"
#include "engine_prediction.hpp"
#include "animations.hpp"
#include "lagcomp.hpp"
#include "movement.hpp"
#include "game_movement.hpp"
#include "entlistener.hpp"
#include "exploits.hpp"
#include "event_logs.hpp"
#include "chams.hpp"
#include "anti_aim.hpp"
#include "penetration.hpp"
#include "resolver.hpp"
#include "ragebot.hpp"
#include "knifebot.hpp"
#include "poly.hpp"

/*
#ifndef _DEBUG
#include <VirtualizerSDK.h>
#endif
*/
	
void draw_hitbox__(c_cs_player* player, matrix3x4_t* bones, int idx, int idx2, bool dur = false)
{
	auto studio_model = HACKS->model_info->get_studio_model(player->get_model());
	if (!studio_model)
		return;

	auto hitbox_set = studio_model->hitbox_set(0);
	if (!hitbox_set)
		return;

	for (int i = 0; i < hitbox_set->num_hitboxes; i++)
	{
		auto hitbox = hitbox_set->hitbox(i);
		if (!hitbox)
			continue;

		vec3_t vMin, vMax;
		math::vector_transform(hitbox->min, bones[hitbox->bone], vMin);
		math::vector_transform(hitbox->max, bones[hitbox->bone], vMax);

		if (hitbox->radius != -1.f)
			HACKS->debug_overlay->add_capsule_overlay(vMin, vMax, hitbox->radius, 255, 255 * idx, 255 * idx2, 150, dur ? HACKS->global_vars->interval_per_tick * 2 : 5.f, 0, 1);
	}
}

INLINE bool valid_hitgroup(int index)
{
	if ((index >= HITGROUP_HEAD && index <= HITGROUP_RIGHTLEG) || index == HITGROUP_GEAR)
		return true;

	return false;
}

bool can_hit_hitbox(const vec3_t& start, const vec3_t& end, rage_player_t* rage, int hitbox, matrix3x4_t* matrix, anim_record_t* record)
{
	auto model = rage->player->get_model();
	if (!model)
		return false;

	auto studio_model = HACKS->model_info->get_studio_model(rage->player->get_model());
	auto set = studio_model->hitbox_set(0);

	if (!set)
		return false;

	auto studio_box = set->hitbox(hitbox);
	if (!studio_box)
		return false;

	vec3_t min, max;

	math::vector_transform(studio_box->min, matrix[studio_box->bone], min);
	math::vector_transform(studio_box->max, matrix[studio_box->bone], max);

	if (studio_box->radius != -1.f)
		return math::segment_to_segment(start, end, min, max) < studio_box->radius;

	c_game_trace trace{};
	HACKS->engine_trace->clip_ray_to_entity({ start, end }, MASK_SHOT_HULL | CONTENTS_HITBOX, rage->player, &trace);

	if (auto ent = trace.entity; ent)
	{
		if (ent == rage->player)
		{
			if (valid_hitgroup(trace.hitgroup))
				return true;
		}
	}

	return false;
}

bool c_ragebot::can_fire(bool ignore_revolver)
{
	if (!HACKS->local || !HACKS->weapon)
		return false;

	if (HACKS->cmd->weapon_select != 0)
		return false;

	if (!HACKS->weapon_info)
		return false;

	if (HACKS->local->flags().has(FL_ATCONTROLS))
		return false;

	if (HACKS->local->wait_for_no_attack())
		return false;

	if (HACKS->local->is_defusing())
		return false;

	if (HACKS->weapon_info->weapon_type >= 1 && HACKS->weapon_info->weapon_type <= 6 && HACKS->weapon->clip1() < 1)
		return false;

	if (HACKS->local->player_state() > 0)
		return false;

	auto weapon_index = HACKS->weapon->item_definition_index();
	if ((weapon_index == WEAPON_GLOCK || weapon_index == WEAPON_FAMAS) && HACKS->weapon->burst_shots_remaining() > 0)
		return HACKS->predicted_time >= HACKS->weapon->next_burst_shot();

	// TO-DO: auto revolver detection
	if (weapon_index == WEAPON_REVOLVER && !ignore_revolver)
		return revolver_fire;

	float next_attack = HACKS->local->next_attack();
	float next_primary_attack = HACKS->weapon->next_primary_attack();

	return HACKS->predicted_time >= next_attack && HACKS->predicted_time >= next_primary_attack;
}

bool c_ragebot::is_shooting()
{
	if (!HACKS->weapon)
		return false;

	bool attack2 = HACKS->cmd->buttons.has(IN_ATTACK2);
	bool attack = HACKS->cmd->buttons.has(IN_ATTACK);

	short weapon_index = HACKS->weapon->item_definition_index();

	if (weapon_index == WEAPON_C4)
		return false;

	if ((weapon_index == WEAPON_GLOCK || weapon_index == WEAPON_FAMAS) && HACKS->weapon->burst_shots_remaining() > 0)
		return HACKS->predicted_time >= HACKS->weapon->next_burst_shot();

	if (HACKS->weapon->is_grenade())
		return !HACKS->weapon->pin_pulled() && HACKS->weapon->throw_time() > 0.f && HACKS->weapon->throw_time() < HACKS->predicted_time;

	auto can_fire_now = can_fire();
	if (weapon_index == WEAPON_REVOLVER)
		return attack && can_fire_now;

	if (HACKS->weapon->is_knife())
		return (attack || attack2) && can_fire_now;

	return attack && can_fire_now;
}

void c_ragebot::update_hitboxes()
{
	if (HACKS->weapon->is_taser())
	{
		hitboxes.emplace_back(HITBOX_STOMACH);
		hitboxes.emplace_back(HITBOX_PELVIS);
		return;
	}

	if (rage_config.hitboxes & head)
		hitboxes.emplace_back(HITBOX_HEAD);

	if (rage_config.hitboxes & chest)
		hitboxes.emplace_back(HITBOX_CHEST);

	if (rage_config.hitboxes & stomach)
		hitboxes.emplace_back(HITBOX_STOMACH);

	if (rage_config.hitboxes & pelvis)
		hitboxes.emplace_back(HITBOX_PELVIS);

	if (rage_config.hitboxes & arms_)
	{
		hitboxes.emplace_back(HITBOX_LEFT_UPPER_ARM);
		hitboxes.emplace_back(HITBOX_RIGHT_UPPER_ARM);
	}

	if (rage_config.hitboxes & legs)
	{
		hitboxes.emplace_back(HITBOX_LEFT_FOOT);
		hitboxes.emplace_back(HITBOX_RIGHT_FOOT);
	}
}

void c_ragebot::run_stop()
{
	if (!HACKS->weapon || HACKS->client_state->delta_tick == -1)
		return;

	if (!HACKS->weapon_info || !HACKS->weapon || !trigger_stop)
		return;

	auto force_acc = rage_config.quick_stop_options & force_accuracy;

	auto max_speed = HACKS->local->is_scoped() ? HACKS->weapon_info->max_speed_alt : HACKS->weapon_info->max_speed;
	max_speed *= 0.34f;

	auto velocity = HACKS->local->velocity();
	if (velocity.length_2d() <= 15.f)
		return;

	if (velocity.length_2d() <= max_speed)
	{
		game_movement::modify_move(*HACKS->cmd, velocity, max_speed);

		if (force_acc)
			should_shot = false;

		return;
	}

	game_movement::force_stop();

	if (force_acc)
		should_shot = true;
}

void c_ragebot::auto_pistol()
{
	if (!HACKS->weapon)
		return;

	auto index = HACKS->weapon->item_definition_index();
	if (index == WEAPON_C4
		|| index == WEAPON_HEALTHSHOT
		|| index == WEAPON_REVOLVER
		|| (index == WEAPON_GLOCK || index == WEAPON_FAMAS) && HACKS->weapon->burst_shots_remaining() > 0)
		return;

	if (HACKS->weapon->is_misc_weapon() && !HACKS->weapon->is_knife())
		return;

	auto next_attack = HACKS->local->next_attack();
	auto next_primary_attack = HACKS->weapon->next_primary_attack();
	auto next_secondary_attack = HACKS->weapon->next_secondary_attack();

	if (HACKS->predicted_time < next_attack || HACKS->predicted_time < next_primary_attack)
	{
		if (HACKS->cmd->buttons.has(IN_ATTACK))
			HACKS->cmd->buttons.remove(IN_ATTACK);
	}

	if (HACKS->predicted_time < next_secondary_attack)
	{
		if (HACKS->cmd->buttons.has(IN_ATTACK2))
			HACKS->cmd->buttons.remove(IN_ATTACK2);
	}
}

void c_ragebot::force_scope()
{
	if (!rage_config.auto_scope)
		return;

	bool able_to_zoom = HACKS->predicted_time >= HACKS->weapon->next_secondary_attack();
	if (able_to_zoom && HACKS->weapon->zoom_level() < 1 && HACKS->weapon->is_sniper() && !HACKS->cmd->buttons.has(IN_ATTACK2))
		HACKS->cmd->buttons.force(IN_ATTACK2);
}


bool c_ragebot::should_stop(const rage_point_t& point)
{
	if (HACKS->weapon->is_taser())
		return false;

	auto unpredicted_vars = ENGINE_PREDICTION->get_unpredicted_vars();
	if (!unpredicted_vars)
		return false;

	if (!(rage_config.quick_stop_options & between_shots) && !can_fire())
		return false;

	if (!MOVEMENT->on_ground() && !(rage_config.quick_stop_options & in_air))
		return false;

	return true;
}

void c_ragebot::update_predicted_eye_pos()
{
	auto unpredicted_vars = ENGINE_PREDICTION->get_initial_vars();
	if (!unpredicted_vars)
		return;

	auto anim = ANIMFIX->get_local_anims();
	if (!anim)
		return;

	auto max_speed = HACKS->local->is_scoped() ? HACKS->weapon_info->max_speed_alt : HACKS->weapon_info->max_speed;
	auto velocity = unpredicted_vars->velocity;

	auto speed = std::max<float>(velocity.length_2d(), 1.f);
	auto max_stop_ticks = std::max<int>(((speed / max_speed) * 5) - 1, 0);

	auto max_predict_ticks = std::clamp(max_stop_ticks, 0, 14);
	if (max_predict_ticks == 0)
	{
		predicted_eye_pos = anim->eye_pos;
		return;
	}

	auto last_predicted_velocity = velocity;
	for (int i = 0; i < max_predict_ticks; ++i)
	{
		auto pred_velocity = velocity * TICKS_TO_TIME(i + 1);

		auto origin = anim->eye_pos + pred_velocity;
		auto flags = HACKS->local->flags();

		game_movement::extrapolate(HACKS->local, origin, pred_velocity, flags, flags.has(FL_ONGROUND));

		last_predicted_velocity = pred_velocity;
	}

	predicted_eye_pos = anim->eye_pos + last_predicted_velocity;
}

void c_ragebot::prepare_players_for_scan()
{
	LISTENER_ENTITY->for_each_player([&](c_cs_player* player)
	{
		auto& rage = rage_players[player->index()];

		if (!player->is_alive() || player->dormant() || player->has_gun_game_immunity())
		{
			if (rage.valid)
				rage.reset();

			return;
		}

		if (rage.player != player)
		{
			rage.reset();
			rage.player = player;
			return;
		}

		rage.distance = HACKS->local->origin().dist_to(player->origin());
		rage.valid = true;

		++rage_player_iter;
	});
}

std::vector<rage_point_t> get_hitbox_points(int damage, std::vector<int>& hitboxes, vec3_t& eye_pos, vec3_t& predicted_eye_pos, rage_player_t* rage, anim_record_t* record, bool predicted = false)
{
	if (hitboxes.empty())
		return {};

	std::vector<rage_point_t> out{};
	out.reserve(hitboxes.size());

	auto backup_origin = HACKS->local->get_abs_origin();

	if (predicted)
		HACKS->local->set_abs_origin(predicted_eye_pos);

	auto matrix_to_aim = record->extrapolated ? record->predicted_matrix : record->matrix_orig.matrix;
	LAGCOMP->set_record(rage->player, record, matrix_to_aim);

	auto local_anims = ANIMFIX->get_local_anims();
	auto& start_eye_pos = predicted ? predicted_eye_pos : local_anims->eye_pos;
	
	int wrong_damage_counter = 0;
	for (auto& hitbox : hitboxes)
	{
		auto aim_point = rage->player->get_hitbox_position(hitbox, matrix_to_aim);
		auto bullet = penetration::simulate(HACKS->local, rage->player, start_eye_pos, aim_point);

		if (bullet.traced_target == nullptr
			|| bullet.traced_target != rage->player
			|| HACKS->weapon->is_taser() && bullet.penetration_count < 4)
			continue;

		rage_point_t point{};
		point.center = true;
		point.hitbox = hitbox;
		point.damage = bullet.damage;
		point.aim_point = aim_point;
		point.predicted_eye_pos = predicted;

#ifndef LEGACY
		point.safety = [&]()
		{
			auto safety = 0;

			matrix3x4_t* matrices[]
			{
				record->matrix_left.matrix,
				record->matrix_left.roll_matrix,
				record->matrix_right.matrix,
				record->matrix_right.roll_matrix,
				record->matrix_zero.matrix,
			};

			for (int i = 0; i < 5; ++i)
			{
				if (can_hit_hitbox(start_eye_pos, aim_point, rage, hitbox, matrices[i], record))
					++safety;
			}

			return safety;
		}();
#endif

		out.emplace_back(point);
	}

	if (predicted)
		HACKS->local->set_abs_origin(backup_origin);

	return out;
}

void player_move(c_cs_player* player, anim_record_t* record)
{
	vec3_t start_pos = record->prediction.origin;
	vec3_t end_pos = start_pos + record->prediction.velocity * HACKS->global_vars->interval_per_tick;

	c_game_trace  tr;
	c_trace_filter filter;
	filter.skip = player;

	HACKS->engine_trace->trace_ray(ray_t(start_pos, end_pos, record->mins, record->maxs), MASK_PLAYERSOLID, &filter, &tr);

	if (tr.fraction != 1.f)
	{
		for (int i = 0; i < 2; ++i)
		{
			if (record->prediction.velocity.length() == 0.f)
				break;

			record->prediction.velocity -= tr.plane.normal * record->prediction.velocity.dot(tr.plane.normal);

			float adjust = record->prediction.velocity.dot(tr.plane.normal);
			if (adjust < 0.f)
				record->prediction.velocity -= tr.plane.normal * adjust;

			start_pos = tr.end;
			end_pos = start_pos + record->prediction.velocity * TICKS_TO_TIME(1.f - tr.fraction);

			HACKS->engine_trace->trace_ray(ray_t(start_pos, end_pos, record->mins, record->maxs), MASK_PLAYERSOLID, &filter, &tr);

			if (tr.fraction == 1.f)
				break;
		}
	}

	record->prediction.origin = tr.end;

	start_pos = tr.end;
	end_pos = start_pos;
	end_pos.z -= 2.f;

	HACKS->engine_trace->trace_ray(ray_t(start_pos, end_pos, record->mins, record->maxs), MASK_PLAYERSOLID, &filter, &tr);

	record->prediction.flags.remove(FL_ONGROUND);

	if (tr.fraction != 1.f && tr.plane.normal.z > 0.7f)
		record->prediction.flags.force(FL_ONGROUND);

	if (record->prediction.flags.has(FL_ONGROUND)) 
	{
		record->layers[4].cycle = 0.f;
		record->layers[4].weight = 0.f;
	}
}

bool start_fakelag_fix(c_cs_player* player, anims_t* anims)
{
	if (anims->records.empty())
		return false;

	if (player->dormant())
		return false;

	size_t size{};
	for (const auto& it : anims->records)
	{
		if (it.dormant)
			break;

		++size;
	}

	auto record = &anims->records.front();
	record->extrapolated = false;
	record->predict();

	if (record->choke <= 0)
		return false;

	if (size > 1 && ((record->origin - anims->records[1].origin).length_sqr() > 4096.f
		|| size > 2 && (anims->records[1].origin - anims->records[2].origin).length_sqr() > 4096.f))
		record->break_lc = true;

	if (!record->break_lc)
		return false;

	int simulation = TIME_TO_TICKS(record->sim_time);
	if (std::abs(HACKS->arrival_tick - simulation) >= 128)
		return false;

	int lag = record->choke;

	int updatedelta = HACKS->client_state->clock_drift_mgr.server_tick - record->server_tick_estimation;
	if (TIME_TO_TICKS(HACKS->outgoing) <= lag - updatedelta)
		return false;

	int next = record->server_tick_estimation + 1;
	if (next + lag >= HACKS->arrival_tick)
		return false;

	auto latency = std::clamp(TICKS_TO_TIME(HACKS->ping), 0.0f, 1.0f);
	auto correct = std::clamp(latency + HACKS->lerp_time, 0.0f, HACKS->convars.sv_maxunlag->get_float());
	auto delta_time = correct - (TICKS_TO_TIME(HACKS->tickbase) - record->sim_time);
	auto predicted_tick = ((int)HACKS->client_state->clock_drift_mgr.server_tick + TIME_TO_TICKS(latency) - record->server_tick_estimation) / record->choke;

	if (predicted_tick > 0 && predicted_tick < 20)
	{
		auto max_backtrack_time = std::ceil(((delta_time - 0.2f) / HACKS->global_vars->interval_per_tick + 0.5f) / (float)record->choke);
		auto prediction_ticks = predicted_tick;

		if (max_backtrack_time > 0.0f && predicted_tick >= TIME_TO_TICKS(max_backtrack_time))
			prediction_ticks = TIME_TO_TICKS(max_backtrack_time);

		if (prediction_ticks > 0)
		{
			record->extrapolate_ticks = prediction_ticks;

			do
			{
				for (auto current_prediction_tick = 0; current_prediction_tick < record->choke; ++current_prediction_tick)
				{
					if (record->prediction.flags.has(FL_ONGROUND))
					{
						if (!HACKS->convars.sv_enablebunnyhopping->get_int())
						{
							float max = player->max_speed() * 1.1f;
							float speed = record->prediction.velocity.length();
							if (max > 0.f && speed > max)
								record->prediction.velocity *= (max / speed);
						}

						record->prediction.velocity.z = HACKS->convars.sv_jump_impulse->get_float();
					}
					else
						record->prediction.velocity.z -= HACKS->convars.sv_gravity->get_float() * HACKS->global_vars->interval_per_tick;

					player_move(player, record);
					record->prediction.time += HACKS->global_vars->interval_per_tick;
				}

				--prediction_ticks;
			} while (prediction_ticks);

			auto current_origin = record->prediction.origin;

			clamp_bones_info_t info{};
			info.collision_change_origin = record->collision_change_origin;
			info.collision_change_time = record->collision_change_time;
			info.origin = current_origin;
			info.collision_origin = current_origin;
			info.ground_entity = record->prediction.flags.has(FL_ONGROUND) ? 1 : -1;
			info.view_offset = record->view_offset;

			math::change_bones_position(record->matrix_orig.matrix, 128, record->origin, current_origin);
			math::memcpy_sse(record->predicted_matrix, record->matrix_orig.matrix, sizeof(record->matrix_orig.matrix));
			record->matrix_orig.bone_builder.clamp_bones_in_bbox(player, record->predicted_matrix, 0x7FF00, record->prediction.time, player->eye_angles(), info);

			math::change_bones_position(record->matrix_orig.matrix, 128, current_origin, record->origin);
			record->extrapolated = true;

			return true;
		}
	}

	return false;
}

multipoints_t c_ragebot::get_points(c_cs_player* player, int hitbox, matrix3x4_t* matrix)
{
	/*
	 * 这个函数早晚是要重写的
	 */

	multipoints_t out;
	if (!player || !player->is_alive())
		return out;

	auto hdr = HACKS->model_info->get_studio_model(player->get_model());
	if (!hdr)
		return out;

	auto set = hdr->hitbox_set(player->hitbox_set());
	if (!set)
		return out;

	auto bbox = set->hitbox(hitbox);
	if (!bbox)
		return out;

	auto build = [&](matrix3x4_t* base, bool has_pred, matrix3x4_t* pred)
		{
			if (!base)
				return;

			vec3_t bbmin, bbmax;
			math::vector_transform(bbox->min, base[bbox->bone], bbmin);
			math::vector_transform(bbox->max, base[bbox->bone], bbmax);

			vec3_t bbmin_p = bbmin, bbmax_p = bbmax;
			if (has_pred && pred)
			{
				math::vector_transform(bbox->min, pred[bbox->bone], bbmin_p);
				math::vector_transform(bbox->max, pred[bbox->bone], bbmax_p);
			}

			vec3_t center = (bbmin + bbmax) * 0.5f;

			if (bbox->radius <= 0.f)
			{
				out.emplace_back(center, true);
				return;
			}

			auto local_anim = ANIMFIX->get_local_anims();
			vec3_t eye_pos = local_anim ? local_anim->eye_pos : HACKS->local->origin();
			vec3_t n = (center - eye_pos).normalized();
			vec3_t u = n.cross({ 0.f, 0.f, 1.f });

			if (u.length_sqr() < 1e-6f)
				u = { 1.f, 0.f, 0.f };

			u.normalized();
			vec3_t v = u.cross(n).normalized();

			auto lerp = [&](const vec3_t& a, const vec3_t& b, float t)
				{
					return (a * (1.f - t) + b * t).normalized() * bbox->radius;
				};

			auto push = [&](const vec3_t& dir, const vec3_t& mn, const vec3_t& mx, std::vector<vec3_t>& ring)
				{
					ring.emplace_back(mn + dir);
					ring.emplace_back(mx + dir);
				};

			std::vector<vec3_t> ring, ring_pred;
			ring.reserve(24);
			ring_pred.reserve(24);

			vec3_t right = u * bbox->radius;
			vec3_t top = vec3_t(0.f, 0.f, 1.f) * bbox->radius;
			vec3_t left = -right;
			vec3_t bot = -top;

			auto fill_ring = [&](std::vector<vec3_t>& r, const vec3_t& mn, const vec3_t& mx)
				{
					push(right, mn, mx, r);
					push(top, mn, mx, r);
					push(left, mn, mx, r);
					push(bot, mn, mx, r);

					push(lerp(right, top, 0.375f), mn, mx, r);
					push(lerp(right, top, 0.625f), mn, mx, r);
					push(lerp(right, bot, 0.375f), mn, mx, r);
					push(lerp(right, bot, 0.625f), mn, mx, r);

					push(lerp(left, top, 0.375f), mn, mx, r);
					push(lerp(left, top, 0.625f), mn, mx, r);
					push(lerp(left, bot, 0.375f), mn, mx, r);
					push(lerp(left, bot, 0.625f), mn, mx, r);
				};

			fill_ring(ring, bbmin, bbmax);
			if (has_pred)
				fill_ring(ring_pred, bbmin_p, bbmax_p);

			for (auto& p : ring)
				p -= n * (p - center).dot(n);
			for (auto& p : ring_pred)
				p -= n * (p - center).dot(n);

			Vector p0(center.x, center.y, center.z);
			if (!ring.empty())
				p0 = { ring.front().x, ring.front().y, ring.front().z };

			std::vector<Vector> flat, flat_pred;
			flat.reserve(ring.size());
			flat_pred.reserve(ring_pred.size());

			auto project = [&](const vec3_t& p)
				{
					Vector q;
					q.x = (p - vec3_t(p0.x, p0.y, p0.z)).dot(u);
					q.y = (p - vec3_t(p0.x, p0.y, p0.z)).dot(v);
					q.z = 0.f;
					return q;
				};

			for (auto& p : ring)
				flat.emplace_back(project(p));

			if (has_pred)
				for (auto& p : ring_pred)
					flat_pred.emplace_back(project(p));

			poly_intersect::graham_scan(flat);
			if (has_pred)
				poly_intersect::graham_scan(flat_pred);

			float rs = 0.975f;

			if (hitbox == HITBOX_LEFT_CALF || hitbox == HITBOX_RIGHT_CALF || hitbox == HITBOX_LEFT_FOOT || hitbox == HITBOX_RIGHT_FOOT)
				rs *= 0.8f; // <-- 如果hitbox靠近人体边缘，则进一步缩小rs

			auto net = ENGINE_PREDICTION->get_networked_vars(HACKS->cmd->command_number); // <-- 网络条件
			rs = std::clamp(rs - (net->spread + net->inaccuracy) * 0.1f, 0.f, 0.975f);

			float scale = hitbox == HITBOX_HEAD ? get_head_scale(player) : get_body_scale(player); // <== 来自菜单的大手
			rs *= 0.5f + 0.5f * std::clamp(scale, 0.f, 0.95f);

			if (!HACKS->convars.cl_lagcompensation->get_int() || !HACKS->convars.cl_predict->get_int())
				rs *= 0.8f; // <-- 我不知道这两个条件有什么用，看fatality也有我也跟着抄了

			auto to_world = [&](const Vector& q)
				{
					return vec3_t(p0.x, p0.y, p0.z) + u * q.x + v * q.y;
				};

			auto emit = [&](std::vector<Vector>& poly)
				{
					if (poly.size() < 3)
						return;

					Vector L = poly[0], R = poly[0], T = poly[0], B = poly[0];
					for (auto& p : poly)
					{
						if (p.x < L.x) L = p;
						if (p.x > R.x) R = p;
						if (p.y > T.y) T = p;
						if (p.y < B.y) B = p;
					}

					Vector c((L.x + R.x + T.x + B.x) * 0.25f, (L.y + R.y + T.y + B.y) * 0.25f, 0.f);

					auto shrink = [&](const Vector& e)
						{
							return Vector(c.x + (e.x - c.x) * rs, c.y + (e.y - c.y) * rs, 0.f);
						};

					Vector ts = shrink(T), bs = shrink(B), ls = shrink(L), rs_ = shrink(R);

					vec3_t center_mod = center;

					if (hitbox == HITBOX_HEAD)
					{
						vec3_t tw = to_world(ts);
						vec3_t bw = to_world(bs);
						vec3_t pt = tw;

						for (int i = 0; i < 6; ++i)
						{
							float t = float(i) / 6.f;
							pt = tw + (bw - tw) * t;
							auto bullet = penetration::simulate(HACKS->local, player, eye_pos, pt);
							if (!bullet.traced_target || bullet.traced_target->index() != player->index() || bullet.hitgroup != HITGROUP_HEAD)
								break;
						}
						center_mod = (tw + pt) * 0.5f;
					}

					out.emplace_back(center_mod, true);

					if (rs >= 0.01f)
					{
						if (hitbox != HITBOX_HEAD)
						{
							out.emplace_back(to_world(ls), false);
							out.emplace_back(to_world(rs_), false);
							out.emplace_back(to_world(bs), false);
						}
						out.emplace_back(to_world(ts), false);
					}
				};

			std::vector<Vector> poly = flat;
			if (has_pred)
				poly = poly_intersect::get_intersection_poly(poly, flat_pred);

			emit(poly);
		};

	if (matrix)
	{
		build(matrix, false, nullptr);
	}
	else
	{
		if (auto anims = ANIMFIX->get_anims(player->index()); anims && !anims->records.empty())
		{
			for (auto& rec : anims->records)
			{
				/* 遍历动画 */
				matrix3x4_t* base = rec.matrix_orig.matrix;
				bool has_pred = rec.extrapolated;
				matrix3x4_t* pred = has_pred ? rec.predicted_matrix : nullptr;
				build(base, has_pred, pred);
			}
		}
	}

	return out;
}

void pre_cache_centers(int damage, std::vector<int>& hitboxes, vec3_t& predicted_eye_pos, rage_player_t* rage)
{
    rage->reset_hitscan();

    auto anims = ANIMFIX->get_local_anims();
    auto lagcomp = ANIMFIX->get_anims(rage->player->index());
    if (!lagcomp || lagcomp->records.empty())
        return;

    auto netchannel = HACKS->engine->get_net_channel();
    if (!netchannel)
        return;

	bool extended_track = g_cfg.rage.extend_track;

    auto get_overall_damage = [&](anim_record_t* record) -> int
        {
            rage->points_to_scan.clear();
            rage->points_to_scan.reserve(MAX_SCANNED_POINTS);

            auto points = get_hitbox_points(damage, hitboxes, predicted_eye_pos, predicted_eye_pos, rage, record, true);
            if (points.empty()) 
                points = get_hitbox_points(damage, hitboxes, anims->eye_pos, predicted_eye_pos, rage, record, false);

            int overall_damage = 0;
            for (const auto& point : points) 
            {
				/* force head */
                //bool safe_headshot = RAGEBOT->rage_config.body_aim_conditions & safe_point_headshot;
                //if (safe_headshot && point.hitbox != HITBOX_HEAD)
                //    continue;

                rage->points_to_scan.emplace_back(point);
                overall_damage += point.damage;
            }

            return overall_damage;
        };

    anim_record_t* best = nullptr;
    anim_record_t* best_candidate = nullptr;
    anim_record_t* valid_record = nullptr;

    rage->restore.store(rage->player);

    if (!HACKS->convars.cl_lagcompensation)
    {
        if (extended_track && lagcomp->records.size() > 1)
        {
            int max_damage = 0;

            for (auto& record : lagcomp->records)
            {
                if (record.valid_lc)
                {
                    int record_damage = get_overall_damage(&record);
                    if (record_damage > max_damage)
                    {
                        max_damage = record_damage;
                        best_candidate = &record;
                    }
                }
            }

            if (best_candidate && max_damage >= damage)
                best = best_candidate;
        }

        anim_record_t* extrapolated_record = &lagcomp->records.back();
        if (start_fakelag_fix(rage->player, lagcomp) && extrapolated_record && get_overall_damage(extrapolated_record) >= damage)
            best = extrapolated_record;
        else 
        {
            auto first_valid = std::find_if(lagcomp->records.begin(), lagcomp->records.end(), [&](anim_record_t& record) {
                return record.valid_lc && get_overall_damage(&record) >= damage;
                });

            if (first_valid != lagcomp->records.end())
                best = &(*first_valid);
        }
    }
    else
    {
        auto last_find = std::find_if(lagcomp->records.rbegin(), lagcomp->records.rend(), [&](anim_record_t& record) {
            return record.valid_lc && get_overall_damage(&record) >= damage;
            });

        if (last_find != lagcomp->records.rend())
            best = &*last_find;
    }

    rage->restore.restore(rage->player);

    if (best)
    {
        rage->start_scans = true;
        rage->hitscan_record = best;
    }
}

void get_result(bool& out, const vec3_t& start, const vec3_t& end, rage_player_t* rage, int hitbox, matrix3x4_t* matrix, anim_record_t* record)
{
	out = can_hit_hitbox(start, end, rage, hitbox, matrix, record);
}

bool hitchance(vec3_t eye_pos, rage_player_t& rage, const rage_point_t& point, anim_record_t* record, const float& chance, matrix3x4_t* matrix, float* hitchance_out = nullptr)
{
	static auto weapon_accuracy_nospread = HACKS->convars.weapon_accuracy_nospread;

	if (weapon_accuracy_nospread && weapon_accuracy_nospread->get_bool())
	{
		if (hitchance_out) *hitchance_out = 1.f;
		return true;
	}

#ifdef LEGACY
	if (EXPLOITS->enabled() && EXPLOITS->dt_bullet == 1)
	{
		if (hitchance_out) *hitchance_out = 1.f;
		return true;
	}
#endif

	auto net_vars = ENGINE_PREDICTION->get_networked_vars(HACKS->cmd->command_number);
	float inaccuracy = net_vars->inaccuracy + net_vars->spread;

	if ((HACKS->ideal_inaccuracy + 0.0005f) >= net_vars->inaccuracy)
	{
		if (hitchance_out) *hitchance_out = 1.f;
		return true;
	}

	auto matrix_to_aim = record->extrapolated ? record->predicted_matrix : record->matrix_orig.matrix;
	auto active_matrix = matrix ? matrix : matrix_to_aim;

	rage.restore.store(rage.player);
	LAGCOMP->set_record(rage.player, record, active_matrix);

	vec3_t aim_dir = (point.aim_point - eye_pos).normalized();
	float dist = eye_pos.dist_to(point.aim_point);
	float spread_r = dist * inaccuracy;

	float tgt_r = 0.f;
	vec3_t tgt_center{};

	if (auto hdr = HACKS->model_info->get_studio_model(rage.player->get_model()))
	{
		if (auto set = hdr->hitbox_set(0))
		{
			if (auto box = set->hitbox(point.hitbox))
			{
				vec3_t mn, mx;
				math::vector_transform(box->min, active_matrix[box->bone], mn);
				math::vector_transform(box->max, active_matrix[box->bone], mx);
				tgt_center = (mn + mx) * 0.5f;
				tgt_r = box->radius > 0.f ? box->radius : (mx - mn).length() * 0.5f;
			}
		}
	}

	vec3_t diff = tgt_center - point.aim_point;
	float d = (diff - aim_dir * diff.dot(aim_dir)).length();

	// 几何解析方法 (lambda)
	auto overlap_ratio = [&](float r1, float r2, float sep) -> float
		{
			if (r1 <= 0.f)
				return 0.f;

			if (sep >= r1 + r2)
				return 0.f;

			if (sep <= std::fabs(r1 - r2))
				return r2 <= r1 ? (r2 * r2) / (r1 * r1) : 1.f;

			float r1sq = r1 * r1, r2sq = r2 * r2;
			float alpha = std::acos((sep * sep + r1sq - r2sq) / (2.f * sep * r1));
			float beta = std::acos((sep * sep + r2sq - r1sq) / (2.f * sep * r2));
			float part = -sep + r1 + r2;
			float area = r1sq * alpha + r2sq * beta - 0.5f * std::sqrt(part * (sep + r1 - r2) * (sep - r1 + r2) * (sep + r1 + r2));

			return area / (M_PI * r1sq);
		};

	float probability = overlap_ratio(spread_r, tgt_r, d);

	if (hitchance_out)
		*hitchance_out = std::clamp(probability, 0.f, 1.f);

	rage.restore.restore(rage.player);

	return probability >= chance;
}


void collect_damage_from_multipoints(int damage, vec3_t& predicted_eye_pos, rage_player_t* rage, rage_point_t& points, anim_record_t* record, matrix3x4_t* matrix_to_aim, bool predicted)
{
	auto multipoints = RAGEBOT->get_points(rage->player, points.hitbox, matrix_to_aim);
	if (multipoints.empty())
		return;

	auto local_anims = ANIMFIX->get_local_anims();

	int wrong_damage_counter = 0;

	auto& start_eye_pos = predicted ? predicted_eye_pos : local_anims->eye_pos;

	auto backup_origin = HACKS->local->get_abs_origin();

	if (predicted)
		HACKS->local->set_abs_origin({ predicted_eye_pos.x, predicted_eye_pos.y, backup_origin.z });

	for (auto& multipoint : multipoints)
	{
		if (multipoint.second)
			continue;

		bool quit_from_scan = false;
		for (auto& i : rage->points_to_scan)
		{
			if (multipoint.first == i.aim_point)
			{
				quit_from_scan = true;
				break;
			}
		}

		if (quit_from_scan)
			break;

		auto bullet = penetration::simulate(HACKS->local, rage->player, start_eye_pos, multipoint.first);
		if (bullet.damage < damage
			|| bullet.traced_target == nullptr || bullet.traced_target != rage->player
			|| HACKS->weapon->is_taser() && bullet.penetration_count < 4)
			continue;

		rage_point_t point{};
		point.center = false;
		point.hitbox = points.hitbox;
		point.damage = bullet.damage;
		point.aim_point = multipoint.first;
		point.predicted_eye_pos = points.predicted_eye_pos;

#ifndef LEGACY
		point.safety = [&]()
		{
			auto safety = 0;

			matrix3x4_t* matrices[]
			{
				record->matrix_left.matrix,
				record->matrix_left.roll_matrix,
				record->matrix_right.matrix,
				record->matrix_right.roll_matrix,
				record->matrix_zero.matrix,
			};

			for (int i = 0; i < 5; ++i)
			{
				if (can_hit_hitbox(start_eye_pos, multipoint.first, rage, points.hitbox, matrices[i], record))
					++safety;
			}

			return safety;
		}();
#endif
		rage->points_to_scan.emplace_back(point);
	}

	if (predicted)
		HACKS->local->set_abs_origin(backup_origin);
}

void c_ragebot::do_hitscan(rage_player_t* rage)
{
	if (rage->points_to_scan.empty())
		return;

	if (!rage->start_scans)
		return;

	if (!rage->hitscan_record)
		return;

	rage->restore.store(rage->player);
	{
		auto matrix_to_aim = rage->hitscan_record->extrapolated ? rage->hitscan_record->predicted_matrix : rage->hitscan_record->matrix_orig.matrix;
		LAGCOMP->set_record(rage->player, rage->hitscan_record, matrix_to_aim);
	
		int threads_count = 0;

		for (auto& points : rage->points_to_scan)
		{
			++threads_count;

			auto dmg = get_min_damage(rage->player);
			THREAD_POOL->add_task(collect_damage_from_multipoints,
				dmg,
				std::ref(predicted_eye_pos),
				rage,
				std::ref(points),
				rage->hitscan_record,
				matrix_to_aim,
				points.predicted_eye_pos);
		}

		if (threads_count > 0)
			THREAD_POOL->wait_all();
	}
	rage->restore.restore(rage->player);
}

void c_ragebot::scan_players()
{
	int threads_count = 0;

	LISTENER_ENTITY->for_each_player([&](c_cs_player* player)
	{
		if (!player->is_alive() || player->dormant() || player->has_gun_game_immunity())
			return;

		auto rage = &rage_players[player->index()];
		if (!rage || !rage->player || rage->player != player)
			return;

		++threads_count;

		auto dmg = get_min_damage(rage->player);
		THREAD_POOL->add_task(pre_cache_centers, dmg, std::ref(hitboxes), std::ref(predicted_eye_pos), rage);
	});

	if (threads_count < 1)
		return;

	THREAD_POOL->wait_all();

	LISTENER_ENTITY->for_each_player([&](c_cs_player* player)
	{
		if (!player->is_alive() || player->dormant() || player->has_gun_game_immunity())
			return;

		auto rage = &rage_players[player->index()];
		if (!rage || !rage->player || rage->player != player)
			return;

		do_hitscan(rage);
	});
}

rage_player_t* c_ragebot::select_target()
{
	static int cycle_idx = 0;
	static int cycle_hits = 0;

	rage_player_t* best{};
	float best_distance = FLT_MAX;
	int best_damage = -1;
	int best_health = 101;
	float best_hitchance_val = -1.f;

	std::vector<rage_player_t*> candidates;
	candidates.reserve(64);

	LISTENER_ENTITY->for_each_player([&](c_cs_player* player)
		{
			if (!player->is_alive() || player->dormant() || player->has_gun_game_immunity())
				return;

			auto rage = &rage_players[player->index()];
			if (!rage || rage->player != player)
				return;

			if (!rage->start_scans || !rage->best_point.found)
				return;

			candidates.emplace_back(rage);

			switch (g_cfg.rage.target_selection)
			{
			case 0: /* lowest distance */

				if (rage->distance < best_distance)
				{
					best_distance = rage->distance;
					best = rage;
				}

				break;

			case 1: /* highest damage */

				if (rage->best_point.damage > best_damage)
				{
					best_damage = rage->best_point.damage;
					best = rage;
				}

				break;

			case 2: /* lowest health */
			{
				int hp = player->health();

				// Tie-breaking
				if (hp < best_health || (hp == best_health && (rage->best_point.damage > best_damage || (rage->best_point.damage == best_damage && rage->distance < best_distance))))
				{
					best_health = hp;
					best_distance = rage->distance; 
					best_damage = rage->best_point.damage;
					best = rage;
				}

				break;
			}

			case 5: /* Best hitchance */
			{
				if (!rage->best_record)
					break;

				auto local_anims = ANIMFIX->get_local_anims();
				if (!local_anims)
					break;

				auto aim_angle = math::calc_angle(local_anims->eye_pos, rage->best_point.aim_point).normalized_angle();
				auto eye_pos_for_hc = ANIMFIX->get_eye_position(aim_angle.x);

				float current_target_hitchance = 0.f;
				float chance_threshold = rage_config.hitchance * 0.01f;

				// calc hitchance for each player
				hitchance(eye_pos_for_hc, *rage, rage->best_point, rage->best_record, chance_threshold, nullptr, &current_target_hitchance);

				// Tie-Breaking
				if (current_target_hitchance > best_hitchance_val ||
					(current_target_hitchance == best_hitchance_val && (rage->best_point.damage > best_damage || (rage->best_point.damage == best_damage && rage->distance < best_distance))))
				{
					best_hitchance_val = current_target_hitchance;
					best_damage = rage->best_point.damage;
					best_distance = rage->distance;
					best = rage;
				}

				break;
			}
			}
		});

	/* 
	 * Cycle & Cycle [2x]
	 */
	if (g_cfg.rage.target_selection == 3 || g_cfg.rage.target_selection == 4)
	{
		if (candidates.empty())
			return nullptr;

		// Sort the player index
		std::sort(candidates.begin(), candidates.end(), [](rage_player_t* a, rage_player_t* b)
			{
				return a->distance < b->distance;
			});

		if (cycle_idx >= static_cast<int>(candidates.size()))
		{
			cycle_idx = 0;
			cycle_hits = 0;
		}

		best = candidates[cycle_idx];

		/*  Cycle */
		if (g_cfg.rage.target_selection == 3)
		{
			cycle_idx = (cycle_idx + 1) % candidates.size();
		}
		else /* Cycle 2x */
		{
			++cycle_hits;

			if (cycle_hits >= 2) // <-- IQ
			{
				cycle_hits = 0;
				cycle_idx = (cycle_idx + 1) % candidates.size();
			}
		}
	}

	return best;
}


void c_ragebot::choose_best_point()
{
	auto prefer_baim_on_dt = EXPLOITS->enabled() && EXPLOITS->get_exploit_mode() == EXPLOITS_DT
		&& (HACKS->weapon->is_auto_sniper() || HACKS->weapon->is_heavy_pistols());

	LISTENER_ENTITY->for_each_player([&](c_cs_player* player)
	{
		if (!player->is_alive() || player->dormant() || player->has_gun_game_immunity())
			return;

		auto rage = &rage_players[player->index()];
		if (!rage || !rage->player || rage->player != player)
			return;

		auto damage = get_min_damage(rage->player);
		auto get_best_aim_point = [&]() -> rage_point_t
		{
			rage_point_t best{};
			std::sort(rage->points_to_scan.begin(), rage->points_to_scan.end(), [](const rage_point_t& a, const rage_point_t& b) {
				return a.damage > b.damage;
			});

			for (auto& point : rage->points_to_scan)
			{
				auto is_body = point.hitbox == HITBOX_PELVIS || point.hitbox == HITBOX_STOMACH;

				if (point.damage < damage)
					continue;
				
				if (g_cfg.binds[force_body_b].toggled && !is_body)
					continue;

#ifndef LEGACY
				if (g_cfg.binds[force_sp_b].toggled && point.safety < 5)
					continue;
#endif
				if (point.safety == 5 && rage_config.prefer_safe)
				{
					point.found = true;
					return point;
				}
				else if (is_body && (point.damage >= player->health() || prefer_baim_on_dt || rage_config.prefer_body))
				{
					point.found = true;
					return point;
				}
				else
				{
					if (point.damage > best.damage)
					{
						best = point;
						best.found = true;
					}
				}
			}

			return best;
		};

		auto best_point = get_best_aim_point();
		if (best_point.found)
		{
			rage->best_point = best_point;
			rage->best_record = rage->hitscan_record;
			rage->best_point.found = true;
		}
	});
}

void c_ragebot::auto_revolver()
{
	if (!HACKS->local || !HACKS->weapon || !HACKS->weapon_info)
		return;

	auto next_secondary_attack = HACKS->weapon->next_secondary_attack();

	if (!g_cfg.rage.enable || EXPLOITS->recharge.start && !EXPLOITS->recharge.finish || HACKS->weapon->item_definition_index() != WEAPON_REVOLVER || HACKS->weapon->clip1() <= 0)
	{
		last_checked = 0;
		tick_cocked = 0;
		tick_strip = 0;
		next_secondary_attack = 0.f;

		revolver_fire = false;
		return;
	}

	auto time = TICKS_TO_TIME(HACKS->tickbase - EXPLOITS->tickbase_offset());
	const auto max_ticks = TIME_TO_TICKS(.25f) - 1;
	const auto tick_base = TIME_TO_TICKS(time);

	if (HACKS->local->next_attack() > time)
		return;

	if (HACKS->local->spawn_time() != last_spawn_time)
	{
		tick_cocked = tick_base;
		tick_strip = tick_base - max_ticks - 1;
		last_spawn_time = HACKS->local->spawn_time();
	}

	if (HACKS->weapon->next_primary_attack() > time)
	{
		HACKS->cmd->buttons.remove(IN_ATTACK);
		revolver_fire = false;
		return;
	}

	if (last_checked == tick_base)
		return;

	last_checked = tick_base;
	revolver_fire = false;

	if (tick_base - tick_strip > 2 && tick_base - tick_strip < 14)
		revolver_fire = true;

	if (HACKS->cmd->buttons.has(IN_ATTACK) && revolver_fire)
		return;

	HACKS->cmd->buttons.force(IN_ATTACK);

	if (next_secondary_attack >= time)
		HACKS->cmd->buttons.force(IN_ATTACK2);

	if (tick_base - tick_cocked > max_ticks * 2 + 1)
	{
		tick_cocked = tick_base;
		tick_strip = tick_base - max_ticks - 1;
	}

	const auto cock_limit = tick_base - tick_cocked >= max_ticks;
	const auto after_strip = tick_base - tick_strip <= max_ticks;

	if (cock_limit || after_strip)
	{
		tick_cocked = tick_base;
		HACKS->cmd->buttons.remove(IN_ATTACK);

		if (cock_limit)
			tick_strip = tick_base;
	}
}

void c_ragebot::run()
{
	if (!HACKS->weapon || !HACKS->weapon_info || HACKS->client_state->delta_tick == -1)
		return;

	auto_pistol();

	if (EXPLOITS->cl_move.trigger && EXPLOITS->cl_move.shifting)
		return;

	hitboxes.clear();
	hitboxes.reserve(HITBOX_MAX);

	rage_config = main_utils::get_weapon_config();
	update_hitboxes();

	trigger_stop = false;
	should_shot = true;
	reset_rage_hitscan = false;
	firing = false;
	working = false;
	rage_player_iter = 0;
	predicted_eye_pos.reset();
	best_rage_player.reset();

	if (!g_cfg.rage.enable || HACKS->weapon->is_misc_weapon() && !HACKS->weapon->is_taser() && !HACKS->weapon->is_knife())
	{
		reset_rage_players();
		return;
	}

	if (HACKS->weapon->is_knife())
	{
		KNIFEBOT->knife_bot();
		return;
	}

	update_predicted_eye_pos();
	prepare_players_for_scan();

	if (rage_player_iter < 1)
	{
		reset_rage_players();
		return;
	}

	reset_rage_hitscan = true;

	scan_players();
	choose_best_point();

	firing = false;
	working = false;
	best_rage_player.reset();

	auto selected_rage = select_target();
	if (selected_rage)
		best_rage_player = *selected_rage;

	const auto& best_point = best_rage_player.best_point;
	if (best_rage_player.player && best_point.found && best_rage_player.start_scans)
	{
		working = true;

		auto local_anims = ANIMFIX->get_local_anims();

		auto damage = get_min_damage(best_rage_player.player);
		bool already_stooped = false;

		if (best_point.predicted_eye_pos && best_point.damage >= damage && (rage_config.quick_stop_options & early))
		{
			force_scope();

			if (rage_config.quick_stop && should_stop(best_point))
			{
				already_stooped = true;
				trigger_stop = true;
			}
		}

		auto aim_angle = math::calc_angle(local_anims->eye_pos, best_point.aim_point).normalized_angle();
		auto ideal_start = ANIMFIX->get_eye_position(aim_angle.x);

		auto best_record = best_rage_player.best_record;

		{
			best_rage_player.restore.store(best_rage_player.player);

			auto matrix_to_aim = best_record->extrapolated ? best_record->predicted_matrix : best_record->matrix_orig.matrix;
			LAGCOMP->set_record(best_rage_player.player, best_record, matrix_to_aim);

			auto final_bullet = penetration::simulate(HACKS->local, best_rage_player.player, ideal_start, best_point.aim_point);
			best_rage_player.restore.restore(best_rage_player.player);

			if (final_bullet.damage < damage || HACKS->weapon->is_taser() && final_bullet.penetration_count < 4)
				return;
		}

		if (!should_shot)
			return;

		if (!already_stooped || !(rage_config.quick_stop_options & early))
		{
			force_scope();

			if (rage_config.quick_stop && should_stop(best_point))
				trigger_stop = true;
		}

		bool supress_doubletap_choke = true;
		if (EXPLOITS->enabled() && EXPLOITS->get_exploit_mode() == EXPLOITS_DT)
			supress_doubletap_choke = EXPLOITS->defensive.tickbase_choke > 2;

		if (!supress_doubletap_choke)
			return;

		if (!can_fire())
			return;

		float out_chance{};
		auto max_hitchance = rage_config.hitchance * 0.01f;

		if (!hitchance(ideal_start, best_rage_player, best_point, best_record, max_hitchance, nullptr, &out_chance))
			return;

		if (g_cfg.rage.auto_fire)
			HACKS->cmd->buttons.force(IN_ATTACK);

		if (HACKS->cmd->buttons.has(IN_ATTACK))
		{
			firing = true;

			auto record_time = best_record->sim_time;

			HACKS->cmd->tickcount = TIME_TO_TICKS(record_time + HACKS->lerp_time);
			auto backtrack_ticks = std::abs(TIME_TO_TICKS(best_rage_player.player->sim_time() - record_time));

			if (g_cfg.visuals.eventlog.logs & 4)
			{
				EVENT_LOGS->push_message(tfm::format(CXOR("Fire to %s [hitbox: %s | hc: %d | sp: %d | dmg: %d | tick: %d]"),
					best_rage_player.player->get_name().c_str(),
					main_utils::hitbox_to_string(best_point.hitbox).c_str(),
					(int)(out_chance * 100.f),
					best_point.safety,
					best_point.damage,
					best_record->extrapolated ? -best_record->extrapolate_ticks : backtrack_ticks), {}, true);
			}

			if (g_cfg.visuals.chams[c_onshot].enable)
				CHAMS->add_shot_record(best_rage_player.player, best_record->matrix_orig.matrix);

			HACKS->cmd->viewangles = math::calc_angle(ideal_start, best_point.aim_point).normalized_angle();
			HACKS->cmd->viewangles -= HACKS->local->aim_punch_angle() * (HACKS->convars.weapon_recoil_scale->get_float());
			HACKS->cmd->viewangles = HACKS->cmd->viewangles.normalized_angle();

			add_shot_record(best_rage_player.player, best_point, best_record, ideal_start);

			if ((g_cfg.binds[hs_b].toggled || !ANTI_AIM->is_fake_ducking()) && !*HACKS->send_packet)
				*HACKS->send_packet = true;
		}
	}

	best_rage_player.reset();
}
void c_ragebot::add_shot_record(c_cs_player* player, const rage_point_t& best, anim_record_t* record, vec3_t eye_pos)
{
	auto anims = ANIMFIX->get_local_anims();

	auto& new_shot = shots.emplace_back();
	new_shot.time = HACKS->predicted_time;
	new_shot.init_time = 0.f;
	new_shot.impact_fire = false;
	new_shot.fire = false;
	new_shot.damage = -1;
	new_shot.safety = best.safety;
	new_shot.start = eye_pos;
	new_shot.hitgroup = -1;
	new_shot.hitchance = best.accuracy;
	new_shot.hitbox = best.hitbox;
	new_shot.pointer = player;
	new_shot.record = *record;
	new_shot.index = player->index();
	new_shot.resolver = resolver_info[new_shot.index];
	new_shot.point = best.aim_point;
}

void c_ragebot::weapon_fire(c_game_event* event)
{
	if (shots.empty())
		return;

	if (HACKS->engine->get_player_for_user_id(event->get_int(CXOR("userid"))) != HACKS->engine->get_local_player())
		return;

	auto& shot = shots.front();
	if (!shot.fire)
		shot.fire = true;
}

void c_ragebot::bullet_impact(c_game_event* event)
{
	if (shots.empty())
		return;

	auto& shot = shots.front();

	if (HACKS->engine->get_player_for_user_id(event->get_int(CXOR("userid"))) != HACKS->engine->get_local_player())
		return;

	const auto vec_impact = vec3_t{ event->get_float(CXOR("x")), event->get_float(CXOR("y")), event->get_float(CXOR("z")) };

	bool check = false;
	if (shot.impact_fire)
	{
		if (shot.start.dist_to(vec_impact) > shot.start.dist_to(shot.impact))
			check = true;
	}
	else
		check = true;

	if (!check)
		return;

	shot.impact_fire = true;
	shot.init_time = HACKS->predicted_time;
	shot.impact = vec_impact;
}

void c_ragebot::player_hurt(c_game_event* event)
{
	if (HACKS->engine->get_player_for_user_id(event->get_int(CXOR("attacker"))) != HACKS->engine->get_local_player())
		return;

	if (!shots.empty())
	{
		auto& shot = shots.front();
		shots.erase(shots.begin());
	}
}

void c_ragebot::round_start(c_game_event* event)
{
	for (auto& i : missed_shots)
		i = 0;

	shots.clear();
}

void c_ragebot::on_game_events(c_game_event* event)
{
	auto name = CONST_HASH(event->get_name());

	switch (name)
	{
	case HASH("weapon_fire"):
		weapon_fire(event);
		break;
	case HASH("bullet_impact"):
		bullet_impact(event);
		break;
	case HASH("player_hurt"):
		player_hurt(event);
		break;
	case HASH("round_start"):
		round_start(event);
		break;
	}
}

void c_ragebot::proceed_misses()
{
	if (shots.empty())
		return;

	auto& shot = shots.front();
	if (std::abs(HACKS->predicted_time - shot.time) > 1.f)
	{
		shots.erase(shots.begin());
		return;
	}

	if (shot.init_time != -1.f && shot.index && shot.damage == -1 && shot.fire && shot.impact_fire)
	{
		auto new_player = (c_cs_player*)HACKS->entity_list->get_client_entity(shot.index);
		if (new_player && new_player->is_player() && shot.pointer == new_player)
		{
			const auto studio_model = HACKS->model_info->get_studio_model(new_player->get_model());

			if (studio_model)
			{
				auto& resolver_info = shot.resolver;
				const auto end = shot.impact;

				auto matrix_to_aim = shot.record.extrapolated ? shot.record.predicted_matrix : shot.record.matrix_orig.matrix;

				rage_player_t rage_player{};
				rage_player.player = new_player;
				rage_player.restore.store(new_player);

				LAGCOMP->set_record(new_player, &shot.record, matrix_to_aim);

				if (!can_hit_hitbox(shot.start, end, &rage_player, shot.hitbox, matrix_to_aim, &shot.record))
				{
					float dist = shot.start.dist_to(shot.impact);
					float dist2 = shot.start.dist_to(shot.point);

					if (dist2 > dist)
						EVENT_LOGS->push_message(XOR("Missed shot due to occlusion"));
					else
						EVENT_LOGS->push_message(XOR("Missed shot due to spread"));
				}
				else
				{
					if (new_player->is_alive())
					{
						if (shot.record.extrapolated)
							EVENT_LOGS->push_message(XOR("Missed shot due to extrapolation failure"));
						else if (resolver_info.resolved)
						{
							missed_shots[shot.index]++;
							EVENT_LOGS->push_message(XOR("Missed shot due to resolver"));
						}
						else
							EVENT_LOGS->push_message(XOR("Missed shot due to ?"));
					}
					else
						EVENT_LOGS->push_message(XOR("Missed shot due to death"));
				}

				rage_player.restore.restore(new_player);
			}
		}

		shots.erase(shots.begin());
	}
}