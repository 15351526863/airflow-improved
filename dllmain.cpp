#include "globals.hpp"

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID reserved)
{
	if (reason == DLL_PROCESS_ATTACH)
	{
		std::thread(init_cheat, reserved).detach();

		return TRUE;
	}

	return false;
}