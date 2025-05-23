#include "globals.hpp"
#include "event_logs.hpp"
#include <algorithm>
#include <cstring>

void c_event_logs::on_item_purchase(c_game_event* event)
{
	if (std::strcmp(event->get_name(), CXOR("item_purchase")) || !HACKS->local)
		return;

	if (!(g_cfg.visuals.eventlog.logs & 8))
		return;

	auto userid = event->get_int(CXOR("userid"));
	if (!userid)
		return;

	auto user_id = HACKS->engine->get_player_for_user_id(userid);
	auto player = static_cast<c_cs_player*>(HACKS->entity_list->get_client_entity(user_id));
	if (!player || player->is_teammate())
		return;

	push_message(tfm::format(CXOR("%s bought %s"),
		player->get_name(),
		event->get_string(CXOR("weapon"))));
}

void c_event_logs::on_bomb_plant(c_game_event* event)
{
	const char* event_name = event->get_name();
	if (!(g_cfg.visuals.eventlog.logs & 16))
		return;

	auto userid = event->get_int(CXOR("userid"));
	if (!userid)
		return;

	auto user_id = HACKS->engine->get_player_for_user_id(userid);
	auto player = static_cast<c_cs_player*>(HACKS->entity_list->get_client_entity(user_id));
	if (!player)
		return;

	if (!std::strcmp(event_name, CXOR("bomb_planted")))
		push_message(tfm::format(CXOR("%s is planted the bomb"), player->get_name()));

	if (!std::strcmp(event_name, CXOR("bomb_begindefuse")))
		push_message(tfm::format(CXOR("%s is defusing the bomb"), player->get_name()));
}

void c_event_logs::on_player_hurt(c_game_event* event)
{
	if (!(g_cfg.visuals.eventlog.logs & 1))
		return;

	if (std::strcmp(event->get_name(), CXOR("player_hurt")) || !HACKS->local)
		return;

	auto attacker = HACKS->engine->get_player_for_user_id(event->get_int(CXOR("attacker")));
	if (HACKS->local->index() != attacker)
		return;

	auto user_id = HACKS->engine->get_player_for_user_id(event->get_int(CXOR("userid")));
	auto player = static_cast<c_cs_player*>(HACKS->entity_list->get_client_entity(user_id));
	if (!player || player->is_teammate())
		return;

	auto group = event->get_int(CXOR("hitgroup"));
	auto dmg_health = event->get_int(CXOR("dmg_health"));
	auto health = event->get_int(CXOR("health"));
	auto string_group = main_utils::hitgroup_to_string(group);

	if (group == HITGROUP_GENERIC || group == HITGROUP_GEAR)
	{
		push_message(tfm::format(CXOR("Hit %s for %d (%d remaining)"),
			player->get_name().c_str(),
			dmg_health,
			health));
	}
	else
	{
		push_message(tfm::format(CXOR("Hit %s in the %s for %d (%d remaining)"),
			player->get_name().c_str(),
			string_group.c_str(),
			dmg_health,
			health));
	}
}

void c_event_logs::on_game_events(c_game_event* event)
{
	on_player_hurt(event);
	on_bomb_plant(event);
	on_item_purchase(event);
}

void c_event_logs::filter_console()
{
	HACKS->convars.con_filter_text->fn_change_callbacks.remove_count();
	HACKS->convars.con_filter_enable->fn_change_callbacks.remove_count();

	if (set_console)
	{
		set_console = false;
		HACKS->cvar->find_convar(CXOR("developer"))->set_value(0);
		HACKS->convars.con_filter_enable->set_value(1);
		HACKS->convars.con_filter_text->set_value(CXOR(""));
	}

	auto filter = g_cfg.visuals.eventlog.enable && g_cfg.visuals.eventlog.filter_console;
	if (log_value != filter)
	{
		log_value = filter;
		if (!log_value)
			HACKS->convars.con_filter_text->set_value(CXOR(""));
		else
			HACKS->convars.con_filter_text->set_value(CXOR("IrWL5106TZZKNFPz4P4Gl3pSN?J370f5hi373ZjPg%VOVh6lN"));
	}
}

void c_event_logs::render_logs()
{
	if (!g_cfg.visuals.eventlog.enable)
		return;

	constexpr float font_size = 15.f;
	auto            render_font = RENDER->fonts.eventlog;
	auto            time = HACKS->system_time();

	float x = 10.f;
	float y = 8.f;
	bool  first = true;

	for (auto it = event_logs.begin(); it != event_logs.end();)
	{
		auto& log = *it;

		float time_left = 1.f - std::clamp((time - log.life_time) / 5.f, 0.f, 1.f);
		if (time_left <= 0.5f)
		{
			float f = std::clamp(time_left, 0.f, 0.5f) / 0.5f;
			log.clr.a() = static_cast<std::uint8_t>(f * 255.f);

			if (first && f < 0.2f)
				y -= font_size * (1.f - f / 0.2f);

			if (time_left <= 0.f)
			{
				it = event_logs.erase(it);
				first = false;
				continue;
			}
		}

		RENDER->text(x, y, log.clr, FONT_DROPSHADOW, &render_font, log.message);

		y += font_size;
		first = false;
		++it;
	}
}