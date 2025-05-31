#include "api_def.hpp"
#include "globals.hpp"
#include "clantag.hpp"
#include <lua.hpp>
#include <random>
#include <algorithm>

#include <Windows.h>
#include <ctime>

namespace lua_api::client
{
    int random_float(lua_State* l)
    {
        int argc = lua_gettop(l);
        if (argc < 2 || !lua_isnumber(l, 1) || !lua_isnumber(l, 2))
        {
            lua_pushstring(l, "usage: random_float(min, max)");
            lua_error(l);
            return 0;
        }

        double min = lua_tonumber(l, 1);
        double max = lua_tonumber(l, 2);
        if (max < min)
            std::swap(min, max);

        static std::mt19937 rng{ std::random_device{}() };
        std::uniform_real_distribution<double> dist(min, max);

        lua_pushnumber(l, dist(rng));
        return 1;
    }

    int random_int(lua_State* l)
    {
        int argc = lua_gettop(l);
        if (argc < 2 || !lua_isnumber(l, 1) || !lua_isnumber(l, 2))
        {
            lua_pushstring(l, "usage: random_int(min, max)");
            lua_error(l);
            return 0;
        }

        lua_Integer min = lua_tointeger(l, 1);
        lua_Integer max = lua_tointeger(l, 2);
        if (max < min)
            std::swap(min, max);

        static std::mt19937 rng{ std::random_device{}() };
        std::uniform_int_distribution<lua_Integer> dist(min, max);

        lua_pushinteger(l, dist(rng));
        return 1;
    }

    int key_state(lua_State* l)
    {
        int argc = lua_gettop(l);
        if (argc < 1 || !lua_isnumber(l, 1))
        {
            lua_pushstring(l, "usage: key_state(vk_code)");
            lua_error(l);
            return 0;
        }

        lua_Integer vk = lua_tointeger(l, 1);
        SHORT state = GetAsyncKeyState(static_cast<int>(vk));
        bool pressed = (state & 0x8000) != 0;
        lua_pushboolean(l, pressed);

        return 1;
    }

    int system_time(lua_State* l)
    {
        int argc = lua_gettop(l);
        if (argc != 0)
        {
            lua_pushstring(l, "usage: system_time()");
            lua_error(l);
            return 0;
        }

        SYSTEMTIME st;
        GetLocalTime(&st);
        lua_pushinteger(l, st.wHour);
        lua_pushinteger(l, st.wMinute);
        lua_pushinteger(l, st.wSecond);
        lua_pushinteger(l, st.wMilliseconds);

        return 4;
    }

    int screen_size(lua_State* l)
    {
        int argc = lua_gettop(l);
        if (argc != 0)
        {
            lua_pushstring(l, "usage: screen_size()");
            lua_error(l);
            return 0;
        }
        int width = GetSystemMetrics(SM_CXSCREEN);
        int height = GetSystemMetrics(SM_CYSCREEN);

        lua_pushinteger(l, width);
        lua_pushinteger(l, height);

        return 2;
    }

    int set_clan_tag(lua_State* l)
    {
        int argc = lua_gettop(l);
        const char* tag = "";
        if (argc == 0)
        {
            tag = "";
        }
        else if (argc == 1 && lua_isstring(l, 1))
        {
            tag = lua_tostring(l, 1);
        }
        else
        {
            lua_pushstring(l, "usage: set_clantag(tag)");
            lua_error(l);
            return 0;
        }

        static auto fn = offsets::clantag.cast<void(__fastcall*)(const char*, const char*)>();

        if (!tag || tag[0] == '\0')
            fn("", "");
        else
            fn(tag, tag);

        return 0;
    }
}