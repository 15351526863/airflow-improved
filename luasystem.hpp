#pragma once
#include <sol/sol.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "event_logs.hpp"
#include "api_def.hpp"

class c_lua_system
{
private:
    sol::state lua;

    std::unordered_map< std::string,
        std::vector< sol::protected_function > > event_callbacks;

    static void register_functions(sol::state& lua_state);

    static void push_message_internal(const std::string& msg,
        const c_color& clr = { 255, 255, 255, 255 },
        bool debug = false);

public:
    c_lua_system();

    void initialize();
    void run_file(const std::string& path);

    void add_event_callback(const std::string& event,
        const sol::protected_function& fn);
    void trigger_event(const std::string& event);

    sol::state& state();
};

inline std::unique_ptr<c_lua_system> LUA_SYSTEM;