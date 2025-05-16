#include "globals.hpp"
#include "entlistener.hpp"
#include "exploits.hpp"
#include "lagcomp.hpp"
#include "knifebot.hpp"

bool c_knifebot::knife_is_behind(c_cs_player* player, anim_record_t* record)
{
	auto origin = record ? record->origin : player->get_abs_origin();
	auto abs_angles = record ? record->abs_angles : player->get_abs_angles();

	auto anim = ANIMFIX->get_local_anims();

	vec3_t delta{ origin - anim->eye_pos };
	delta.z = 0.f;
	delta = delta.normalized();

	vec3_t target;
	math::angle_vectors(abs_angles, target);
	target.z = 0.f;

	return delta.dot(target) > 0.475f;
}

bool c_knifebot::knife_trace(vec3_t dir, bool stab, c_game_trace* trace)
{
	float range = stab ? 32.f : 48.f;

	auto anim = ANIMFIX->get_local_anims();

	vec3_t start = anim->eye_pos;
	vec3_t end = start + (dir * range);

	c_trace_filter filter{};
	filter.skip = HACKS->local;
	HACKS->engine_trace->trace_ray(ray_t(start, end), MASK_SOLID, &filter, trace);

	if (trace->fraction >= 1.f)
	{
		HACKS->engine_trace->trace_ray(ray_t(start, end, { -16.f, -16.f, -18.f }, { 16.f, 16.f, 18.f }), MASK_SOLID, &filter, trace);
		return trace->fraction < 1.f;
	}

	return true;
}

bool c_knifebot::can_knife(c_cs_player* player, anim_record_t* record, vec3_t angle, bool& stab)
{
	vec3_t forward{};
	math::angle_vectors(angle, forward);

	c_game_trace trace{};
	knife_trace(forward, false, &trace);

	if (!trace.entity || trace.entity != player)
		return false;

	bool armor = player->armor_value() > 0;
	bool first = HACKS->weapon->next_primary_attack() + 0.4f < HACKS->predicted_time;
	bool back = knife_is_behind(player, record);

	int stab_dmg = knife_dmg.stab[armor][back];
	int slash_dmg = knife_dmg.swing[first][armor][back];
	int swing_dmg = knife_dmg.swing[false][armor][back];

	int health = player->health();
	if (health <= slash_dmg)
		stab = false;
	else if (health <= stab_dmg)
		stab = true;
	else if (health > (slash_dmg + swing_dmg + stab_dmg))
		stab = true;
	else
		stab = false;

	if (stab && !knife_trace(forward, true, &trace))
		return false;

	return true;
}

void c_knifebot::knife_bot()
{
	if (!g_cfg.rage.enable)
		return;

	if (HACKS->predicted_time < HACKS->weapon->next_primary_attack() || HACKS->predicted_time < HACKS->weapon->next_secondary_attack())
		return;

	bool supress_doubletap_choke = true;
	if (EXPLOITS->enabled() && EXPLOITS->get_exploit_mode() == EXPLOITS_DT)
		supress_doubletap_choke = EXPLOITS->defensive.tickbase_choke > 2;

	bool best_stab{};
	knife_point_t best{};

	LISTENER_ENTITY->for_each_player([&](c_cs_player* player)
		{
			if (!player->is_alive() || player->dormant() || player->has_gun_game_immunity())
				return;

			auto anims = ANIMFIX->get_anims(player->index());
			if (!anims || anims->records.empty())
				return;

			auto first_find = std::find_if(anims->records.begin(), anims->records.end(), [&](anim_record_t& record) {
				return record.valid_lc;
				});

			anim_record_t* first = nullptr;
			if (first_find != anims->records.end())
				first = &*first_find;

			restore_record_t backup{};
			backup.store(player);

			if (!first)
			{
				backup.restore(player);
				return;
			}

			{
				{
					LAGCOMP->set_record(player, first, first->matrix_orig.matrix);

					for (auto& a : knife_ang)
					{
						if (!can_knife(player, first, a, best_stab))
							continue;

						best.point = a;
						best.record = first;
						break;
					}
				}

				{
					auto last_find = std::find_if(anims->records.rbegin(), anims->records.rend(), [&](anim_record_t& record) {
						return record.valid_lc;
						});

					anim_record_t* last = nullptr;
					if (last_find != anims->records.rend())
						last = &*last_find;

					if (!last || last == first)
					{
						backup.restore(player);
						return;
					}

					LAGCOMP->set_record(player, last, last->matrix_orig.matrix);

					for (auto& a : knife_ang)
					{
						if (!can_knife(player, last, a, best_stab))
							continue;

						best.point = a;
						best.record = last;
						break;
					}
				}
			}
			backup.restore(player);

			if (best.record)
			{
				backup.restore(player);
				return;
			}
		});

	if (supress_doubletap_choke && best.record)
	{
		HACKS->cmd->viewangles = best.point.normalized_angle();

		if (best.record && !HACKS->cl_lagcomp0)
			HACKS->cmd->tickcount = TIME_TO_TICKS(best.record->sim_time + HACKS->lerp_time);

		HACKS->cmd->buttons.force(best_stab ? IN_ATTACK2 : IN_ATTACK);
	}
}
