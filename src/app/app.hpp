#pragma once

#include "../input/input.hpp"

namespace app
{
	constexpr auto APP_TITLE = "GPU Game";

	// allocate memory
	constexpr u32 SCREEN_BUFFER_HEIGHT = 800;
	constexpr u32 SCREEN_BUFFER_WIDTH = SCREEN_BUFFER_HEIGHT * 10 / 8;


	class AppMemory
	{
	public:
		b32 is_app_initialized;
		
		size_t permanent_storage_size;
		void* permanent_storage;
	};


	class  ScreenBuffer
	{
	public:
		void* memory;
		u32 width;
		u32 height;
		u32 bytes_per_pixel;
	};


	// app.cpp
	bool initialize_memory(AppMemory& memory, ScreenBuffer& buffer);

	// app.cpp
	void update_and_render(AppMemory& memory, Input const& input);

	// app.cpp
	void end_program(AppMemory& memory);
}


void platform_signal_stop();