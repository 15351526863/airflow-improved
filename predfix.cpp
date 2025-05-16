#include "globals.hpp"
#include "predfix.hpp"
#include "event_logs.hpp"

// fix all wrong values for predicted netvars
void c_prediction_fix::store(int tick)
{
    auto netvars = &compressed_netvars[tick % 150];
    netvars->store(tick);
}

void c_prediction_fix::fix_netvars(int tick)
{
    pred_error_occured = false;

    auto netvars = &compressed_netvars[tick % 150];
    if (!netvars->filled || netvars->cmd_number != tick)
        return;

    if (netvars->tickbase != 0 && HACKS->local->tickbase() != netvars->tickbase)
        return;

    //if ( checks )
    //      EVENT_LOGS->info_push_message(XOR("The information from the server didn't come true on the client, prediction error occured"), { 255, 99, 99 });

    auto aim_punch_diff = netvars->aimpunch - HACKS->local->aim_punch_angle();
    if (std::abs(aim_punch_diff.x) <= 0.03125f && std::abs(aim_punch_diff.y) <= 0.03125 && std::abs(aim_punch_diff.z) <= 0.03125f)
        HACKS->local->aim_punch_angle() = netvars->aimpunch;
    else
        pred_error_occured = true;

    auto view_punch_diff = netvars->viewpunch - HACKS->local->view_punch_angle();
    if (std::abs(view_punch_diff.x) <= 0.03125f && std::abs(view_punch_diff.y) <= 0.03125f && std::abs(view_punch_diff.z) <= 0.03125f)
        HACKS->local->view_punch_angle() = netvars->viewpunch;
    else
        pred_error_occured = true;

    auto aim_punch_vel_diff = netvars->aimpunch_vel - HACKS->local->aim_punch_angle_vel();
    if (std::abs(aim_punch_vel_diff.x) <= 0.03125f && std::abs(aim_punch_vel_diff.y) <= 0.03125 && std::abs(aim_punch_vel_diff.z) <= 0.03125f)
        HACKS->local->aim_punch_angle_vel() = netvars->aimpunch_vel;
    else
        pred_error_occured = true;

    auto view_offset_diff = netvars->viewoffset - HACKS->local->view_offset();
    if (std::abs(view_offset_diff.z) <= 0.065f)
        HACKS->local->view_offset() = netvars->viewoffset;
    else
        pred_error_occured = true;

    auto origin_diff = netvars->origin - HACKS->local->origin();
    if (std::abs(origin_diff.x) < 0.03125f && std::abs(origin_diff.y) < 0.03125f && std::abs(origin_diff.z) < 0.03125f)
        HACKS->local->origin() = netvars->origin;
    else
        pred_error_occured = true;

    auto fall_velocity_diff = netvars->fall_velocity - HACKS->local->fall_velocity();
    if (std::abs(fall_velocity_diff) <= 0.5f)
        HACKS->local->fall_velocity() = netvars->fall_velocity;
    else
        pred_error_occured = true;

    auto crouch_amount_diff = netvars->duck_amount - HACKS->local->duck_amount();
    if (std::abs(crouch_amount_diff) < 0.03125f)
    {
        HACKS->local->duck_amount() = netvars->duck_amount;
        HACKS->local->duck_speed() = netvars->duck_speed;
    }
    else
        pred_error_occured = true;

    auto velocity_modifier_diff = netvars->velocity_modifier - HACKS->local->velocity_modifier();
    if (std::abs(velocity_modifier_diff) < 0.03125f)
        HACKS->local->velocity_modifier() = netvars->velocity_modifier;
    else
        pred_error_occured = true;

    auto base_velocity_diff = netvars->base_velocity - HACKS->local->base_velocity();
    if (std::abs(base_velocity_diff.x) < 0.03125f && std::abs(base_velocity_diff.y) < 0.03125f && std::abs(base_velocity_diff.z) < 0.03125f)
        HACKS->local->base_velocity() = netvars->base_velocity;
    else
        pred_error_occured = true;

    auto velocity_diff = netvars->velocity - HACKS->local->velocity();
    if (std::abs(velocity_diff.x) > 0.5f || std::abs(velocity_diff.y) > 0.5f || std::abs(velocity_diff.z) > 0.5f)
        pred_error_occured = true;

    auto net_origin_diff = netvars->network_origin - HACKS->local->network_origin();
    if (std::abs(net_origin_diff.x) > 0.0625f || std::abs(net_origin_diff.y) > 0.0625f || std::abs(net_origin_diff.z) > 0.0625f)
        pred_error_occured = true;

    if (!pred_error_occured)
        return;

    HACKS->prediction->prev_start_frame = -1;
    HACKS->prediction->commands_predicted = 0;
    HACKS->prediction->prev_ack_had_errors = 1;
}