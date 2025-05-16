#pragma once
#include "resolver.hpp"
#include "globals.hpp"

struct knife_point_t
{
	int damage{};
	vec3_t point{};
	anim_record_t* record{};
};

class c_knifebot
{
private:
	struct table_t
	{
		std::uint8_t swing[2][2][2]{};
		std::uint8_t stab[2][2]{};
	};

	const table_t knife_dmg{ { { { 25, 90 }, { 21, 76 } }, { { 40, 90 }, { 34, 76 } } }, { { 65, 180 }, { 55, 153 } } };
	
	std::array<vec3_t, 12 > knife_ang
	{
		vec3_t{ 0.f, 0.f, 0.f },
		vec3_t{ 0.f, -90.f, 0.f },
		vec3_t{ 0.f, 90.f, 0.f },
		vec3_t{ 0.f, 180.f, 0.f },
		vec3_t{ -80.f, 0.f, 0.f },
		vec3_t{ -80.f, -90.f, 0.f },
		vec3_t{ -80.f, 90.f, 0.f },
		vec3_t{ -80.f, 180.f, 0.f },
		vec3_t{ 80.f, 0.f, 0.f },
		vec3_t{ 80.f, -90.f, 0.f },
		vec3_t{ 80.f, 90.f, 0.f },
		vec3_t{ 80.f, 180.f, 0.f }
	};

public:
	bool knife_is_behind(c_cs_player* player, anim_record_t* record);
	bool knife_trace(vec3_t dir, bool stab, c_game_trace* trace);
	bool can_knife(c_cs_player* player, anim_record_t* record, vec3_t angle, bool& stab);
	void knife_bot();
};

#ifndef _DEBUG
inline auto KNIFEBOT = std::make_unique<c_knifebot>();
#endif