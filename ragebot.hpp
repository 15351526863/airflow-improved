#pragma once
#include "lagcomp.hpp"
#include "resolver.hpp"

constexpr int MAX_SCANNED_POINTS = 150;
constexpr float MATRIX_HEAD_ROTATION = 0.70710678f;
constexpr int MAX_SEEDS = 128;

using multipoint_t = std::pair<vec3_t, bool>;
using multipoints_t = std::vector<multipoint_t>;
using rage_bullet_t = std::pair<vec3_t, std::uint64_t>;

struct resolver_info_t;
inline std::vector<std::tuple<float, float, float>> precomputed_seeds{};

struct restore_record_t;

struct aim_shot_record_t
{
	bool fire{};
	bool impact_fire{};

	int safety{};
	int damage{};
	int hitgroup{};
	int hitbox{};

	float time{};
	float init_time{};
	float hitchance{};

	int index{};

	vec3_t start{};
	vec3_t impact{};
	vec3_t point{};

	c_cs_player* pointer{};

	anim_record_t record{};
	resolver_info_t resolver{};
};

struct rage_point_t
{
	bool found = false;
	bool center = false;
	bool extrapolated = false;
	bool predicted_eye_pos = false;

	int hitbox = 0;
	int damage = -1;
	int safety = 0;

	float accuracy = 0.f;
	float priority = 0.f;

	vec3_t aim_point{};

	INLINE void reset()
	{
		found = false;
		center = false;
		extrapolated = false;
		predicted_eye_pos = false;

		hitbox = 0;
		damage = 0;
		safety = 0;

		accuracy = 0.f;
		priority = 0.f;

		aim_point.reset();
	}
};

struct rage_player_t
{
	bool valid = false;
	bool start_scans = false;

	float distance{};
	c_cs_player* player{};

	restore_record_t restore{};

	anim_record_t* hitscan_record{};
	anim_record_t* best_record{};

	rage_point_t best_point{};

	std::vector<rage_point_t> points_to_scan{};

	INLINE rage_player_t() {}
	INLINE rage_player_t(c_cs_player* player) : player(player) {
		distance = HACKS->local->origin().dist_to(player->origin());
	}

	INLINE void reset_hitscan()
	{
		start_scans = false;
		hitscan_record = nullptr;
		best_record = nullptr;

		best_point = {};

		points_to_scan.clear();
		points_to_scan.reserve(MAX_SCANNED_POINTS);
	}

	INLINE void reset()
	{
		valid = false;
		start_scans = false;

		distance = 0.f;
		player = nullptr;

		restore.reset();

		hitscan_record = nullptr;
		best_record = nullptr;

		best_point.reset();

		points_to_scan.clear();
	}
};

struct knife_point_t
{
        int damage{};
        vec3_t point{};
        anim_record_t* record{};
};

struct PlayerState_t
{
        int m_iSomeFlag{};
};

struct AimContext_t
{
        int m_iCurrentBestHitbox{};
        c_cs_player* m_pCurrentBestTarget{};
        int m_iTargetPlayerIndex{};
        int m_iLastBestHitbox{};
        bool m_bHasValidTarget{};
        float m_flSomeTimestamp{};
        int m_iSomeTargetID1{};
        bool m_bIgnoreBounds{};
        int m_iLastScannedHitbox{};
        vec3_t m_vecLastScannedPoint{};
        bool m_bAimbotHasTarget{};
        c_cs_player* m_pFinalTarget{};
        int m_iFinalHitbox{};
        float m_flFinalDamage{};
        vec3_t m_vecFinalPoint{};

        INLINE bool IsPointWithinBounds(const vec3_t&) const { return true; }
        INLINE bool IsPointWithinSecondaryBounds(const vec3_t&) const { return true; }
        INLINE void SetPointAsInvalid(int) {}
        INLINE void ClearFailureFlags() {}
        INLINE void UpdateTargetPoint(const vec3_t&) {}
        INLINE bool ShouldOverrideTarget(c_cs_player*) const { return true; }
};

PlayerState_t* GetPlayerState(c_cs_player* player);
bool Ragebot_RegisterAimPoint(AimContext_t* pAimContext, const vec3_t& vecPoint,
        c_cs_player* pTargetEntity, int iHitbox);


class c_ragebot
{
private:
	bool should_shot = true;
	bool reset_rage_hitscan = false;
	int rage_player_iter = 0;

	int last_checked = 0;
	float last_spawn_time = 0.f;
	int tick_cocked = 0;
	int tick_strip = 0;

	std::vector<int> hitboxes{};

        vec3_t predicted_eye_pos{};
        rage_player_t rage_players[65]{};

        struct aim_point_visual_t
        {
                bool valid{};
                int hitbox{};
                vec3_t point{};

                INLINE void reset()
                {
                        valid = false;
                        hitbox = 0;
                        point.reset();
                }
        };

        aim_point_visual_t aim_points[65]{};

        std::vector<aim_shot_record_t> shots{};

        struct lag_record_t
        {
                c_cs_player* player{};
        };

        std::vector<c_cs_player*> m_activeTargets{};
        std::vector<lag_record_t> m_lagRecords{};

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

	void update_hitboxes();

	INLINE void reset_rage_players()
	{
		if (reset_rage_hitscan)
		{
			for (auto& i : rage_players)
				i.reset();

			reset_rage_hitscan = false;
			rage_player_iter = 0;
		}
	}

	void force_scope();
	bool should_stop(const rage_point_t& point);
	void update_predicted_eye_pos();
	void prepare_players_for_scan();
	void do_hitscan(rage_player_t* rage);
	void scan_players();
	void choose_best_point();
	void add_shot_record(c_cs_player* player, const rage_point_t& best, anim_record_t* record, vec3_t eye_pos);

	void weapon_fire(c_game_event* event);
	void bullet_impact(c_game_event* event);
	void player_hurt(c_game_event* event);
	void round_start(c_game_event* event);

public:
	bool trigger_stop = false;
	bool firing{};
	bool working{};
	bool revolver_fire{};
	int missed_shots[65]{};
	rage_player_t best_rage_player{};
        rage_weapon_t rage_config{};

        INLINE aim_point_visual_t* get_aim_point(int idx)
        {
                return &aim_points[idx];
        }

	INLINE float get_dynamic_scale(const vec3_t& point, const float& hitbox_radius)
	{
		auto anim = ANIMFIX->get_local_anims();
		auto predicted_info = ENGINE_PREDICTION->get_networked_vars(HACKS->cmd->command_number);

		auto spread = predicted_info->spread + predicted_info->inaccuracy;
		auto distance = point.dist_to(anim->eye_pos);

		auto new_dist = distance / std::sin(DEG2RAD(90.f - RAD2DEG(spread)));
		auto scale = (hitbox_radius - new_dist * spread) + 0.1f;

		return std::clamp(scale, 0.f, 0.95f);
	}

	INLINE float get_head_scale(c_cs_player* player)
	{
		if (rage_config.scale_head != -1)
			return rage_config.scale_head * 0.01f;

		return 0.f;
	}

	INLINE float get_body_scale(c_cs_player* player)
	{
		if (rage_config.scale_body != -1)
			return rage_config.scale_body * 0.01f;

		return 0.f;
	}

	INLINE int get_min_damage(c_cs_player* player)
	{
		auto mindamage = g_cfg.binds[override_dmg_b].toggled ?
			rage_config.damage_override : rage_config.mindamage;

		auto health = player->health();
		if (mindamage > 99)
			return health + (mindamage - 100);

		if (mindamage > player->health())
			return health;

		return mindamage;
	}

	INLINE void reset()
	{
		trigger_stop = false;
		should_shot = false;
		reset_rage_hitscan = false;
		firing = false;
		working = false;
		revolver_fire = false;
		rage_player_iter = 0;
		last_checked = 0;
		last_spawn_time = 0.f;
		tick_cocked = 0;
		tick_strip = 0;

		for (auto& i : rage_players)
			i.reset();

		for (auto& i : missed_shots)
			i = 0;

		rage_config = {};
		best_rage_player.reset();
		predicted_eye_pos.reset();
                hitboxes.clear();
                shots.clear();
                m_activeTargets.clear();
                m_lagRecords.clear();

                for (auto& i : aim_points)
                        i.reset();
        }

	INLINE void build_seeds()
	{
		// clear it before build seeds, for some reason :)
		precomputed_seeds.clear();
		precomputed_seeds.reserve(MAX_SEEDS);

		for (auto i = 0; i < MAX_SEEDS; i++)
		{
			math::random_seed(i + 1);

			const auto pi_seed = math::random_float(0.f, M_PI * 2);
			precomputed_seeds.emplace_back(math::random_float(0.f, 1.f), std::sin(pi_seed), std::cos(pi_seed));
		}
	}

	bool can_fire();
	bool is_shooting();
	void run_stop();
	void auto_pistol();
	void auto_revolver();
	bool knife_is_behind(c_cs_player* player, anim_record_t* record);
        bool knife_trace(vec3_t dir, bool stab, c_game_trace* trace);
        bool can_knife(c_cs_player* player, anim_record_t* record, vec3_t angle, bool& stab);
        void knife_bot();
        void UpdateOrPruneTargetHistory(c_cs_player* target);
        bool CheckHitchance(const vec3_t& shoot_pos, c_cs_player* target, c_base_combat_weapon* weapon, const vec3_t& aim_angle, float required);
        void on_game_events(c_game_event* event);
        void proceed_misses();
        void run();
};

#ifdef _DEBUG
inline auto RAGEBOT = std::make_unique<c_ragebot>();
#else
CREATE_DUMMY_PTR(c_ragebot);
DECLARE_XORED_PTR(c_ragebot, GET_XOR_KEYUI32);

#define RAGEBOT XORED_PTR(c_ragebot)
#endif