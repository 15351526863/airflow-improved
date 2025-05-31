#include "../../globals.hpp"
#include "menu.h"
#include "snow.h"

#include "../config_system.h"
#include "../config_vars.h"

#include <ShlObj.h>
#include <algorithm>
#include <map>
#include <d3d9.h>
#include <d3dx9.h>

/*
#ifndef _DEBUG
#include <VirtualizerSDK.h>
#endif // !_DEBUG
*/

#define add_texture_to_memory D3DXCreateTextureFromFileInMemory

std::vector< std::string > key_strings = { XOR("None"), XOR("M1"), XOR("M2"), XOR("Ctrl+brk"), XOR("M3"), XOR("M4"), XOR("M5"), XOR(" "), XOR("Back"), XOR("Tab"),
  XOR(" "), XOR(" "), XOR(" "), XOR("Enter"), XOR(" "), XOR(" "), XOR("Shift"), XOR("Ctrl"), XOR("Alt"), XOR("Pause"), XOR("Caps"), XOR(" "), XOR(" "),
  XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR("Esc"), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR("Spacebar"), XOR("Pgup"), XOR("Pgdwn"), XOR("End"),
  XOR("Home"), XOR("Left"), XOR("Up"), XOR("Right"), XOR("Down"), XOR(" "), XOR("Print"), XOR(" "), XOR("Prtsc"), XOR("Insert"), XOR("Delete"), XOR(" "),
  XOR("0"), XOR("1"), XOR("2"), XOR("3"), XOR("4"), XOR("5"), XOR("6"), XOR("7"), XOR("8"), XOR("9"), XOR(" "), XOR(" "), XOR(" "), XOR(" "),
  XOR(" "), XOR(" "), XOR(" "), XOR("A"), XOR("B"), XOR("C"), XOR("D"), XOR("E"), XOR("F"), XOR("G"), XOR("H"), XOR("I"), XOR("J"), XOR("K"),
  XOR("L"), XOR("M"), XOR("N"), XOR("O"), XOR("P"), XOR("Q"), XOR("R"), XOR("S"), XOR("T"), XOR("U"), XOR("V"), XOR("W"), XOR("X"), XOR("Y"),
  XOR("Z"), XOR("Lw"), XOR("Rw"), XOR(" "), XOR(" "), XOR(" "), XOR("Num 0"), XOR("Num 1"), XOR("Num 2"), XOR("Num 3"), XOR("Num 4"), XOR("Num 5"),
  XOR("Num 6"), XOR("Num 7"), XOR("Num 8"), XOR("Num 9"), XOR("*"), XOR("+"), XOR("_"), XOR("-"), XOR("."), XOR("/"), XOR("F1"), XOR("F2"), XOR("F3"),
  XOR("F4"), XOR("F5"), XOR("F6"), XOR("F7"), XOR("F8"), XOR("F9"), XOR("F10"), XOR("F11"), XOR("F12"), XOR("F13"), XOR("F14"), XOR("F15"), XOR("F16"),
  XOR("F17"), XOR("F18"), XOR("F19"), XOR("F20"), XOR("F21"), XOR("F22"), XOR("F23"), XOR("F24"), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "),
  XOR(" "), XOR(" "), XOR(" "), XOR("Num lock"), XOR("Scroll lock"), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "),
  XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR("Lshift"), XOR("Rshift"), XOR("Lcontrol"), XOR("Rcontrol"), XOR("Lmenu"), XOR("Rmenu"),
  XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR("Next track"), XOR("Previous track"),
  XOR("Stop"), XOR("Play/pause"), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(";"), XOR("+"), XOR("),"), XOR("-"), XOR("."),
  XOR("/?"), XOR("~"), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "),
  XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "),
  XOR("[{"), XOR("\\|"), XOR("}]"), XOR("'\""), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "),
  XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "),
  XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" "), XOR(" ") };

std::array< std::string, max_tabs > tabs = { XOR("Rage"), XOR("Legit"), XOR("Anti-hit"), XOR("Visuals"), XOR("Misc"), XOR("Skins"), XOR("Configs") };

void c_menu::create_animation(float& mod, bool cond, float speed_multiplier, unsigned int animation_flags)
{
	float time = (RENDER->get_animation_speed() * 5.f) * speed_multiplier;

	if ((animation_flags & skip_enable) && cond)
		mod = 1.f;
	else if ((animation_flags & skip_disable) && !cond)
		mod = 0.f;

	if (animation_flags & lerp_animation)
		mod = std::lerp(mod, (float)cond, time);
	else
	{
		if (cond && mod <= 1.f)
			mod += time;
		else if (!cond && mod >= 0.f)
			mod -= time;
	}

	mod = std::clamp(mod, 0.f, 1.f);
}

void c_menu::update_alpha()
{
	this->create_animation(alpha, g_cfg.misc.menu, 0.3f);
}

float c_menu::get_alpha()
{
	return alpha;
}

void c_menu::set_window_pos(const ImVec2& pos)
{
	window_pos = pos;
}

ImVec2 c_menu::get_window_pos()
{
	return window_pos + ImVec2(45, 15);
}

void c_menu::init_textures()
{
	if (!logo_texture)
		add_texture_to_memory(RENDER->get_device(), cheatLogo, sizeof(cheatLogo), &logo_texture);

	if (!keyboard_texture)
		add_texture_to_memory(RENDER->get_device(), keyboard_icon, sizeof(keyboard_icon), &keyboard_texture);

	if (!warning_texture)
		add_texture_to_memory(RENDER->get_device(), warning_icon, sizeof(warning_icon), &warning_texture);

	if (!spectator_texture)
		add_texture_to_memory(RENDER->get_device(), spectators_icon, sizeof(spectators_icon), &spectator_texture);

	if (!bomb_texture)
		add_texture_to_memory(RENDER->get_device(), bomb_indicator, sizeof(bomb_indicator), &bomb_texture);

	if (!icon_textures[0])
		add_texture_to_memory(RENDER->get_device(), rage_icon, sizeof(rage_icon), &icon_textures[0]);

	if (!icon_textures[1])
		add_texture_to_memory(RENDER->get_device(), legit_icon, sizeof(legit_icon), &icon_textures[1]);

	if (!icon_textures[2])
		add_texture_to_memory(RENDER->get_device(), antihit_icon, sizeof(antihit_icon), &icon_textures[2]);

	if (!icon_textures[3])
		add_texture_to_memory(RENDER->get_device(), visuals_icon, sizeof(visuals_icon), &icon_textures[3]);

	if (!icon_textures[4])
		add_texture_to_memory(RENDER->get_device(), misc_icon, sizeof(misc_icon), &icon_textures[4]);

	if (!icon_textures[5])
		add_texture_to_memory(RENDER->get_device(), skins_icon, sizeof(skins_icon), &icon_textures[5]);

	if (!icon_textures[6])
		add_texture_to_memory(RENDER->get_device(), cfg_icon, sizeof(cfg_icon), &icon_textures[6]);

	//MessageBoxA(0, g_cheat_info->user_avatar.c_str(), 0, 0);

	if (!avatar && !HACKS->cheat_info.user_avatar.empty() && HACKS->cheat_info.user_avatar.size())
		add_texture_to_memory(RENDER->get_device(), HACKS->cheat_info.user_avatar.data(), HACKS->cheat_info.user_avatar.size(), &avatar);
}

void c_menu::set_draw_list(ImDrawList* list)
{
	if (draw_list)
		return;

	draw_list = list;
}

ImDrawList* c_menu::get_draw_list()
{
	return draw_list;
}

void c_menu::window_begin()
{
	static bool opened = true;
	const ImVec2 window_size(800.f, 540.f);
	static ImVec2 drag_offset(0.f, 0.f);

	ImGui::SetNextWindowPos(ImVec2(RENDER->screen.x * 0.5f - window_size.x * 0.5f, RENDER->screen.y * 0.5f - window_size.y * 0.5f), ImGuiCond_Once);
	ImGui::SetNextWindowSize(window_size, ImGuiCond_Once);
	ImGui::PushFont(RENDER->fonts.main.get());
	ImGui::SetNextWindowBgAlpha(0.f);
	ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.f, 0.f, 0.f, 0.f));
	ImGui::Begin(CXOR("##base_window"), &opened, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollbar);

	ImGui::PushItemWidth(256.f);

	ImGui::SetCursorPos(ImVec2(0.f, 0.f));
	ImGui::InvisibleButton("##header_block", ImVec2(720.f, 47.f));
	if (ImGui::IsItemActive())
	{
		if (ImGui::IsItemActivated())
			drag_offset = ImGui::GetMousePos() - ImGui::GetWindowPos();
		ImGui::SetWindowPos(ImGui::GetMousePos() - drag_offset);
	}

	ImVec2 saved_cursor = ImGui::GetCursorPos();
	ImGui::SetCursorPos(ImVec2(0.f, 15.f + 520.f));
	ImGui::InvisibleButton("##footer_block", ImVec2(720.f, 20.f));
	if (ImGui::IsItemActive())
	{
		if (ImGui::IsItemActivated())
			drag_offset = ImGui::GetMousePos() - ImGui::GetWindowPos();
		ImGui::SetWindowPos(ImGui::GetMousePos() - drag_offset);
	}
	ImGui::SetCursorPos(saved_cursor);

	this->set_draw_list(ImGui::GetWindowDrawList());
	this->set_window_pos(ImGui::GetWindowPos());

	auto list = this->get_draw_list();
	list->Flags |= ImDrawListFlags_AntiAliasedFill | ImDrawListFlags_AntiAliasedLines;

	ImVec2 window_pos = this->get_window_pos();

	list->PushClipRect(window_pos, ImVec2(window_pos.x + 720.f, window_pos.y + 540.f));
	ImGui::PushClipRect(window_pos, ImVec2(window_pos.x + 720.f, window_pos.y + 540.f), false);
	list->PushClipRect(window_pos, ImVec2(window_pos.x + 720.f, window_pos.y + 550.f));
	ImGui::PushClipRect(window_pos, ImVec2(window_pos.x + 720.f, window_pos.y + 550.f), false);
}


void c_menu::window_end()
{
	auto list = this->get_draw_list();
	list->Flags &= ~(ImDrawListFlags_AntiAliasedFill | ImDrawListFlags_AntiAliasedLines);
	list->PopClipRect();
	ImGui::PopClipRect();

	ImGui::PopItemWidth();

	ImGui::End(false);
	ImGui::PopStyleColor();
	ImGui::PopFont();
}

void c_menu::draw_ui_background()
{
	auto  list = this->get_draw_list();
	float alpha = this->get_alpha();
	if (alpha <= 0.f)
		return;

	constexpr float W = 720.f;   // window width (fixed)
	constexpr float H_HEAD = 47.f;    // header height
	constexpr float H_SHADOW = 8.f;     // header drop-shadow band
	constexpr float H_FOOT = 20.f;    // footer height

	const float a255 = 255.f * alpha;
	ImVec2 pos = this->get_window_pos();

	/* colour helpers */
	const auto accent = g_cfg.misc.ui_color.base().new_alpha(int(a255)).as_imcolor();
	ImColor top_grad = ImColor(35, 35, 35, int(a255));          // header top
	ImColor bot_grad = ImColor(12, 12, 12, int(a255));          // header bottom
	ImColor hi_line = ImColor(70, 70, 70, int(180.f * alpha)); // top bevel
	ImColor lo_line = ImColor(0, 0, 0, int(120.f * alpha)); // bottom groove

	/* ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤ HEADER (material style) ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤ */

	// main plate: subtle vertical gradient
	list->AddRectFilledMultiColor(pos,
		pos + ImVec2(W, H_HEAD),
		top_grad, top_grad, bot_grad, bot_grad);

	// top 1-px highlight for bevel effect
	list->AddRectFilled(pos, pos + ImVec2(W, 1.f), hi_line);

	// bottom 1-px inner shadow groove
	list->AddRectFilled(pos + ImVec2(0.f, H_HEAD - 1.f),
		pos + ImVec2(W, H_HEAD), lo_line);

	// 2-px accent stripe on extreme left
	list->AddRectFilled(pos, pos + ImVec2(2.f, H_HEAD), accent);
	// faint shadow to the right of the stripe to lift it
	list->AddRectFilled(pos + ImVec2(2.f, 0.f),
		pos + ImVec2(3.f, H_HEAD),
		ImColor(0, 0, 0, int(100.f * alpha)));

	/* logo (24¡Á24) */
	const float logo_wh = 24.f;
	ImVec2 img_min = pos + ImVec2(6.f, (H_HEAD - logo_wh) * 0.5f);
	ImVec2 img_max = img_min + ImVec2(logo_wh, logo_wh);
	if (logo_texture)
		list->AddImage((void*)logo_texture, img_min, img_max,
			ImVec2(0, 0), ImVec2(1, 1), accent);

	/* word-mark with soft shadow */
	const std::string brand = "AIRFLOW";
	ImGui::PushFont(RENDER->fonts.bold_large.get()
		? RENDER->fonts.bold_large.get()
		: RENDER->fonts.bold.get());
	ImVec2 txt_sz = ImGui::CalcTextSize(brand.c_str());
	ImVec2 txt_pos = ImVec2(img_max.x + 8.f,
		pos.y + (H_HEAD - txt_sz.y) * 0.5f);
	list->AddText(txt_pos + ImVec2(0.f, 1.f),
		ImColor(0, 0, 0, int(200.f * alpha)), brand.c_str());
	list->AddText(txt_pos, accent, brand.c_str());
	ImGui::PopFont();

	/* drop-shadow band under header (fades to transparent) */
	ImVec2 sh_min = pos + ImVec2(0.f, H_HEAD);
	ImVec2 sh_max = sh_min + ImVec2(W, H_SHADOW);
	list->AddRectFilledMultiColor(sh_min, sh_max,
		ImColor(0, 0, 0, int(150.f * alpha)),
		ImColor(0, 0, 0, int(150.f * alpha)),
		ImColor(0, 0, 0, 0),
		ImColor(0, 0, 0, 0));

	/* ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤ BODY BLUR ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤ */
	imgui_blur::create_blur(list,
		pos + ImVec2(0.f, H_HEAD),
		pos + ImVec2(W, 520.f),
		ImColor(80, 80, 80, int(a255)));

	/* ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤ FOOTER (material style) ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤ */
	ImVec2 foot_pos = pos + ImVec2(0.f, 520.f);
	ImVec2 foot_size = ImVec2(W, H_FOOT);

	// footer gradient
	list->AddRectFilledMultiColor(foot_pos, foot_pos + foot_size,
		ImColor(30, 30, 30, int(a255)),
		ImColor(30, 30, 30, int(a255)),
		ImColor(10, 10, 10, int(a255)),
		ImColor(10, 10, 10, int(a255)));

	// top light bevel & bottom edge
	list->AddRectFilled(foot_pos, foot_pos + ImVec2(W, 1.f),
		ImColor(45, 45, 45, int(a255)));
	list->AddRectFilled(foot_pos + ImVec2(0.f, H_FOOT - 1.f),
		foot_pos + foot_size,
		ImColor(0, 0, 0, int(a255)));

	/* footer text */
	ImGui::PushFont(RENDER->fonts.main.get());
	const char* accent_txt = "@Airflow";
	const char* rest_txt = " Developer: Essentialia";
	ImVec2 a_sz = ImGui::CalcTextSize(accent_txt);
	ImVec2 r_sz = ImGui::CalcTextSize(rest_txt);
	float tx = foot_pos.x + foot_size.x - a_sz.x - r_sz.x - 10.f;
	float ty = foot_pos.y + (H_FOOT - a_sz.y) * 0.5f;
	// subtle shadow
	list->AddText(ImVec2(tx, ty) + ImVec2(0.f, 1.f),
		ImColor(0, 0, 0, int(180.f * alpha)), accent_txt);
	list->AddText(ImVec2(tx + a_sz.x, ty) + ImVec2(0.f, 1.f),
		ImColor(0, 0, 0, int(180.f * alpha)), rest_txt);
	// main
	list->AddText(ImVec2(tx, ty), accent, accent_txt);
	list->AddText(ImVec2(tx + a_sz.x, ty),
		c_color(200, 200, 200, int(150.f * alpha)).as_imcolor(),
		rest_txt);
	ImGui::PopFont();

	/* ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤ DIVIDER & OUTER BORDER ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤ */
	list->AddLine(pos + ImVec2(160.f, H_HEAD),
		pos + ImVec2(160.f, 520.f),
		c_color(255, 255, 255, 12.75f * alpha).as_imcolor());

	list->AddRect(pos + ImVec2(0.f, H_HEAD),
		pos + ImVec2(W, 540.f),
		c_color(100, 100, 100, 100.f * alpha).as_imcolor());
}


void c_menu::draw_tabs()
{
	auto  list = this->get_draw_list();
	ImVec2 prev_cursor = ImGui::GetCursorPos();
	float alpha = this->get_alpha();           // 0 ¡ú 1 (menu fade-in)
	float win_alpha = 255.f * alpha;               // 0 ¡ú 255
	ImVec2 origin = this->get_window_pos() + ImVec2(0.f, 49.f);

	/* Invisible buttons use these four colours */
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
	ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 1.f, alpha));

	/* Geometry constants that exactly match the legacy layout */
	const float  row_height = 40.f;                 // full vertical stride
	const ImVec2 btn_size = ImVec2(144.f, 32.f);  // clickable area
	const float  btn_x = 53.f;                 // cursor X for buttons
	const float  first_y = 78.f;                 // cursor Y for first row
	const ImVec2 draw_offset = ImVec2(8.f, 14.f);    // from origin to draw-rect
	const auto   accent_clr = g_cfg.misc.ui_color.base();

	for (int i = 0; i < tabs.size(); ++i)
	{
		auto& info = tab_info[i];

		/* 1. ImGui invisible button (handles input / focus) */
		ImGui::SetCursorPos(ImVec2(btn_x, first_y + row_height * i));
		std::string id = CXOR("##tab_") + std::to_string(i);
		info.selected = ImGui::ButtonEx(id.c_str(), btn_size, 0, &info.hovered);
		if (info.selected)
			tab_selector = i;

		/* 2. Animate hover-pulse & active alpha */
		create_animation(info.hovered_alpha, info.hovered, 1.f, lerp_animation);
		create_animation(info.alpha, tab_selector == i, 0.8f,
			skip_disable | lerp_animation);

		/* 3. Compute absolute rectangle in draw-list space */
		ImVec2 tab_min = origin + draw_offset + ImVec2(0.f, row_height * i);
		ImVec2 tab_max = tab_min + btn_size;           // width = 144  ( ¡Ü 160-px panel )
		// height = 32
/* 4. Active-tab visuals -------------------------------------------------- */
		if (tab_selector == i)
		{
			/* Left yellow stripe (2-px inside the 160-px panel) */
			list->AddRectFilled(ImVec2(tab_min.x - 2.f, tab_min.y),
				ImVec2(tab_min.x, tab_max.y),
				accent_clr.new_alpha(int(win_alpha * info.alpha))
				.as_imcolor());

			/* Dark selected-row backdrop (tiny alpha so it¡¯s almost black) */
			int sel_a = int(10.f * info.alpha * alpha);   // 0 ¡ú 10
			list->AddRectFilled(tab_min,
				tab_max,
				c_color(255, 255, 255, sel_a).as_imcolor());
		}

		/* 5. Text colour --------------------------------------------------------- */
		c_color txt;
		if (tab_selector == i)
			txt = accent_clr.new_alpha(int(win_alpha));   // bright accent
		else
		{
			float lum = 150.f + 105.f * info.hovered_alpha;      // 150 ¡ú 255
			txt = c_color(lum, lum, lum, lum * alpha);
		}

		/* 6. Render tab label (no icon in legacy style) -------------------------- */
		list->AddText(tab_min + ImVec2(12.f, 8.f), txt.as_imcolor(), tabs[i].c_str());
	}

	ImGui::PopStyleColor(4);
	ImGui::SetCursorPos(prev_cursor);
}


void c_menu::draw_sub_tabs(int& selector, const std::vector< std::string >& tabs)
{
	auto& style = ImGui::GetStyle();
	auto alpha = this->get_alpha();
	auto window_alpha = 255.f * alpha;
	auto child_pos = this->get_window_pos() + ImVec2(178, 62);
	auto prev_pos = ImGui::GetCursorPos();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
	ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 1.f, this->get_alpha()));

	draw_list->AddRectFilled(child_pos, child_pos + ImVec2(528, 58), c_color(217, 217, 217, 20 * alpha).as_imcolor(), 4.f);

	auto clr = g_cfg.misc.ui_color.base();

	for (int i = 0; i < tabs.size(); ++i)
	{
		auto& info = subtab_info[tabs[0]][i];

		const auto cursor_pos = ImVec2(27 + 80 * i, 14);
		ImGui::SetCursorPos(cursor_pos);

		const auto tab_size = ImVec2(70.f, 32.f);

		auto tab_str = CXOR("##sub_tab_") + tabs[i];
		info.selected = ImGui::ButtonEx(tab_str.c_str(), tab_size, 0, &info.hovered);

		if (info.selected)
			selector = i;

		this->create_animation(info.hovered_alpha, info.hovered, 1.f, lerp_animation);
		this->create_animation(info.alpha, selector == i, 0.8f, skip_disable | lerp_animation);

		// idk why but base pos offsets by 8 pixels
		// so move to 8 pixels left for correct pos
		const auto tab_min = child_pos + cursor_pos - ImVec2(8, 0);
		const auto tab_max = tab_min + tab_size;
		const auto tab_bb = ImRect(tab_min, tab_max);

		if (selector == i)
		{
			draw_list->AddRectFilled(tab_bb.Min, tab_bb.Max, c_color(217, 217, 217, 20 * alpha * info.alpha).as_imcolor(), 4.f);

			// i can't draw rounded rect for 2 pixels
			// so i decided to limit render range and draw rect for 4 pixels
			draw_list->PushClipRect(ImVec2(tab_bb.Min.x + 15.f, tab_bb.Max.y - 2.f), ImVec2(tab_bb.Max.x - 15.f, tab_bb.Max.y));
			draw_list->AddRectFilled(ImVec2(tab_bb.Min.x + 15.f, tab_bb.Max.y - 2.f), ImVec2(tab_bb.Max.x - 15.f, tab_bb.Max.y + 4.f), clr.new_alpha(info.alpha * window_alpha).as_imcolor(), 2.f, ImDrawCornerFlags_Top);
			draw_list->PopClipRect();
		}

		float rgb_val = selector == i ? 255 : 150 + 105 * info.hovered_alpha;
		c_color text_clr = c_color(rgb_val, rgb_val, rgb_val, rgb_val * this->get_alpha());

		const ImVec2 label_size = ImGui::CalcTextSize(tabs[i].c_str(), NULL, true);

		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(text_clr.r() / 255.f, text_clr.g() / 255.f, text_clr.b() / 255.f, text_clr.a() / 255.f));
		ImGui::RenderTextClipped(tab_bb.Min, tab_bb.Max, tabs[i].c_str(), NULL, &label_size, style.ButtonTextAlign, &tab_bb);
		ImGui::PopStyleColor();
	}

	ImGui::PopStyleColor();

	ImGui::SetCursorPos(prev_pos);

	// spacing for tab elements
	ImGui::ItemSize(ImVec2(0, 62));

	ImGui::PushClipRect(child_pos + ImVec2(0.f, 62.f), child_pos + ImVec2(540, 457), false);
	draw_list->PushClipRect(child_pos + ImVec2(0.f, 62.f), child_pos + ImVec2(540, 457));
}

std::vector< Snowflake::Snowflake > snow;

void c_menu::draw_snow()
{

}

void c_menu::draw()
{
	this->init_textures();

	if (RENDER->screen.x <= 0.f || RENDER->screen.y <= 0.f)
	{
		g_cfg.misc.watermark_position.x = 0.f;
		g_cfg.misc.watermark_position.y = 0.f;

		g_cfg.misc.keybind_position.x = 0.f;
		g_cfg.misc.keybind_position.y = 0.f;

		g_cfg.misc.bomb_position.x = 0.f;
		g_cfg.misc.bomb_position.y = 0.f;

		g_cfg.misc.spectators_position.x = 0.f;
		g_cfg.misc.spectators_position.y = 0.f;
		return;
	}

	this->update_alpha();
	this->draw_binds();
	this->draw_bomb_indicator();
	this->draw_watermark();
	this->draw_spectators();

	static auto reset = false;
	static auto set_ui_focus = false;

	if (!g_cfg.misc.menu && alpha <= 0.f)
	{
		if (reset)
		{
			for (auto& a : item_animations)
			{
				a.second.reset();
				tab_alpha = 0.f;
				subtab_alpha = 0.f;
				subtab_alpha2 = 0.f;
			}
			reset = false;
		}

		HACKS->loading_config = false;
		set_ui_focus = false;
		return;
	}

	reset = true;

	if (!set_ui_focus)
	{
		ImGui::SetNextWindowFocus();
		set_ui_focus = true;
	}

	ImGui::SetColorEditOptions(ImGuiColorEditFlags_PickerHueBar | ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoOptions | ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_DisplayRGB);

	//this->draw_snow();
	this->window_begin();
	this->draw_ui_background();
	this->draw_tabs();
	this->draw_ui_items();
	this->window_end();

	HACKS->loading_config = false;
}