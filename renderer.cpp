#include "globals.hpp"
#include "api_def.hpp"
#include <sol/sol.hpp>

namespace renderer
{
    int rectangle(lua_State* L)
    {
        int x = sol::stack::get<int>(L, 1);
        int y = sol::stack::get<int>(L, 2);
        int w = sol::stack::get<int>(L, 3);      // width  (not x2)
        int h = sol::stack::get<int>(L, 4);      // height (not y2)
        int r = sol::stack::get<int>(L, 5);
        int g = sol::stack::get<int>(L, 6);
        int b = sol::stack::get<int>(L, 7);
        int a = sol::stack::get<int>(L, 8);

        HACKS->surface->draw_set_color(r, g, b, a);
        HACKS->surface->draw_filled_rect(x, y, x + w, y + h);   // �� convert here

        return 0;
    }

    int line(lua_State* L)
    {
        int xa = sol::stack::get<int>(L, 1);
        int ya = sol::stack::get<int>(L, 2);
        int xb = sol::stack::get<int>(L, 3);
        int yb = sol::stack::get<int>(L, 4);
        int r = sol::stack::get<int>(L, 5);
        int g = sol::stack::get<int>(L, 6);
        int b = sol::stack::get<int>(L, 7);
        int a = sol::stack::get<int>(L, 8);

        HACKS->surface->draw_set_color(r, g, b, a);
        HACKS->surface->draw_line(xa, ya, xb, yb);
        return 0;
    }
}