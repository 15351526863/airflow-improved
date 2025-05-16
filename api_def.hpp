#pragma once
struct lua_State;

namespace lua_api
{
    namespace client
    {
        int random_float(lua_State* l);

        int random_int(lua_State* l);

        int key_state(lua_State* l);

        int system_time(lua_State* l);

        int screen_size(lua_State* l);

        int set_clan_tag(lua_State* l);
    }
}

namespace renderer
{
    int rectangle(lua_State* l);

    int line(lua_State* L);
}