#pragma once

#include "app.hpp"
#include "../gpu/gpu_app.hpp"

#ifndef NDEBUG
#define PRINT_APP_ERROR
#endif

#ifdef PRINT_APP_ERROR
#include <cstdio>
#endif

static void print_error(cstr msg)
{
#ifdef PRINT_APP_ERROR
	printf("\n*** APP ERROR ***\n\n");
	printf("%s", msg);
	printf("\n\n******************\n\n");
#endif
}


constexpr f32 screen_height_m(f32 screen_width_m)
{
    return screen_width_m * SCREEN_HEIGHT_PX / SCREEN_WIDTH_PX;
}


constexpr f32 px_to_m(f32 n_pixels, f32 width_m, u32 width_px)
{
    return n_pixels * width_m / width_px;
}


constexpr size_t host_memory_sz()
{
    // HostImage screen_pixels;
    //auto screen_pixels_data_sz = app::SCREEN_BUFFER_WIDTH * app::SCREEN_BUFFER_HEIGHT * sizeof(Pixel);

    return  0; //screen_pixels_data_sz;
}


bool init_device_memory(AppState& state, app::ScreenBuffer& buffer);

bool init_unified_memory(AppState& state);

void init_app_input(AppInput& app_input);