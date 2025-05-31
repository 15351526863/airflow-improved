#include "globals.hpp"
#include "luasystem.hpp"
#include "api_def.hpp"
#include <unordered_map>
#include <vector>

void c_lua_system::push_message_internal(const std::string& msg,
    const c_color& clr,
    bool debug)
{
    if (debug)
    {
        auto base_clr = g_cfg.misc.ui_color.base();
        HACKS->cvar->print_console_color(base_clr, CXOR("[AIRFLOW] "));
        HACKS->cvar->print_console_color(c_color{ 150, 150, 150 }, "%s\n", msg.c_str());
        return;
    }

    EVENT_LOGS->push_message(msg, clr);
}

void c_lua_system::add_event_callback(const std::string& event, const sol::protected_function& fn)
{
    event_callbacks[event].push_back(fn);
}

void c_lua_system::trigger_event(const std::string& event)
{
    auto it = event_callbacks.find(event);
    if (it == event_callbacks.end())
        return;

    for (auto& cb : it->second)
    {
        if (!cb.valid())
            continue;

        sol::protected_function_result res = cb();
        if (!res.valid())
        {
            sol::error err = res;

            push_message_internal(err.what(),
                { 255, 0, 0, 255 },
                true);
        }
    }
}

void c_lua_system::register_functions(sol::state& lua_state)
{
    lua_state.set_function("print",
        [](sol::this_state ts, sol::variadic_args va)
        {
            lua_State* L = ts;
            std::string out;
            for (auto v : va)
            {
                switch (v.get_type())
                {
                case sol::type::string:
                    out.append(v.as<std::string_view>());
                    break;
                case sol::type::boolean:
                    out.append(v.as<bool>() ? "true" : "false");
                    break;
                case sol::type::number:
                    if (lua_isinteger(L, v.stack_index()))
                        out.append(std::to_string(v.as<lua_Integer>()));
                    else
                        out.append(std::to_string(v.as<double>()));
                    break;
                default:
                    out.append("<unsupported>");
                    break;
                }
                out.push_back(' ');
            }
            if (!out.empty())
                out.pop_back();

            c_lua_system::push_message_internal(out);
        });

    sol::table client_table = lua_state["client"].get_or_create<sol::table>();
    client_table.set_function("random_float", &lua_api::client::random_float);
    client_table.set_function("random_int", &lua_api::client::random_int);
    client_table.set_function("key_state", &lua_api::client::key_state);
    client_table.set_function("system_time", &lua_api::client::system_time);
    client_table.set_function("screen_size", &lua_api::client::screen_size);
    client_table.set_function("set_clan_tag", &lua_api::client::set_clan_tag);

    client_table.set_function("set_event_callback",
        [](const std::string& event, sol::protected_function fn)
        {
            if (LUA_SYSTEM)
                LUA_SYSTEM->add_event_callback(event, fn);
        });

    sol::table renderer_table = lua_state["renderer"].get_or_create<sol::table>();
    renderer_table.set_function("rectangle", &renderer::rectangle);
    renderer_table.set_function("line", &renderer::line);
}

c_lua_system::c_lua_system()
{
    initialize();
}

void c_lua_system::initialize()
{
    lua.open_libraries(sol::lib::base,
        sol::lib::package,
        sol::lib::table,
        sol::lib::string,
        sol::lib::math,
        sol::lib::utf8,
        sol::lib::coroutine);

    register_functions(lua);
}

void c_lua_system::run_file(const std::string& path)
{
    lua.safe_script_file(path,
        [](lua_State*, sol::protected_function_result pfr)
        {
            if (!pfr.valid())
            {
                sol::error err = pfr;
                c_lua_system::push_message_internal(err.what(),
                    { 255, 0, 0, 255 },
                    true);
            }
            return pfr;
        });
}

sol::state& c_lua_system::state()
{
    return lua;
}

static struct lua_system_initializer_t
{
    lua_system_initializer_t()
    {
        LUA_SYSTEM = std::make_unique<c_lua_system>();
    }
} _lua_system_initializer_;