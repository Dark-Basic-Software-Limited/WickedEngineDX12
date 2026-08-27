#pragma once
// This file includes platform, os specific libraries and supplies common platform specific resources

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <SDKDDKVer.h>
#include <windows.h>

#if WINAPI_FAMILY == WINAPI_FAMILY_GAMES
#define PLATFORM_XBOX
#else
#define PLATFORM_WINDOWS_DESKTOP
#endif // WINAPI_FAMILY_GAMES
#define wiLoadLibrary(name) LoadLibraryA(name)
#define wiGetProcAddress(handle,name) GetProcAddress(handle, name)
#elif defined(__SCE__)
#define PLATFORM_PS5
#else
#define PLATFORM_LINUX
#include <dlfcn.h>
#define wiLoadLibrary(name) dlopen(name, RTLD_LAZY)
#define wiGetProcAddress(handle,name) dlsym(handle, name)
typedef void* HMODULE;
#endif // _WIN32

#ifdef SDL2
#include <SDL2/SDL.h>
#include <SDL_vulkan.h>
#include "sdl2.h"
#endif


namespace wi::platform
{
#ifdef _WIN32
	using window_type = HWND;
	using error_type = HRESULT;
#elif defined(SDL2)
	using window_type = SDL_Window*;
	using error_type = int;
#else
	using window_type = void*;
	using error_type = int;
#endif // _WIN32

#ifdef _WIN32
	// ★ GGMAX 3.26. The main thread's id, captured once at startup.
	//
	// PostQuitMessage posts WM_QUIT to the CALLING thread's queue. OnDeviceRemoved can run on a
	// threadpool thread (RegisterWaitForSingleObject), and a pool thread has no message loop - so
	// the quit went nowhere, the main loop never learned to stop, and the app carried on
	// rendering into a dead device until something else killed it. That is the window the 08-27
	// crash happened inside.
	inline unsigned long& main_thread_id() { static unsigned long id = 0; return id; }
	inline void SetMainThread() { main_thread_id() = GetCurrentThreadId(); }
#endif // _WIN32

	inline void Exit()
	{
#ifdef _WIN32
		const unsigned long gg_main = main_thread_id();
		if (gg_main != 0 && gg_main != GetCurrentThreadId())
		{
			// Cross-thread: PostQuitMessage would be a no-op here.
			PostThreadMessage(gg_main, WM_QUIT, 0, 0);
		}
		else
		{
			PostQuitMessage(0);
		}
#endif // _WIN32
#ifdef SDL2
		SDL_Event quit_event;
		quit_event.type = SDL_QUIT;
		SDL_PushEvent(&quit_event);
#endif
	}

	struct WindowProperties
	{
		int width = 0;
		int height = 0;
		float dpi = 96;
	};
	inline void GetWindowProperties(window_type window, WindowProperties* dest)
	{
#ifdef PLATFORM_WINDOWS_DESKTOP
		dest->dpi = (float)GetDpiForWindow(window);
#endif // WINDOWS_DESKTOP

#ifdef PLATFORM_XBOX
		dest->dpi = 96.f;
#endif // PLATFORM_XBOX

#if defined(PLATFORM_WINDOWS_DESKTOP) || defined(PLATFORM_XBOX)
		RECT rect;
		GetClientRect(window, &rect);
		dest->width = int(rect.right - rect.left);
		dest->height = int(rect.bottom - rect.top);
#endif // PLATFORM_WINDOWS_DESKTOP || PLATFORM_XBOX

#ifdef PLATFORM_LINUX
		int window_width, window_height;
		SDL_GetWindowSize(window, &window_width, &window_height);
		SDL_Vulkan_GetDrawableSize(window, &dest->width, &dest->height);
		dest->dpi = ((float)dest->width / (float)window_width) * 96.f;
#endif // PLATFORM_LINUX
	}
}
