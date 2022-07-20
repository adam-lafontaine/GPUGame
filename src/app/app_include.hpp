#pragma once

#include "app.hpp"
#include "gpu/gpu_app.hpp"

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


constexpr auto GRASS_TILE_PATH = "/home/adam/Repos/GPUGame/assets/tiles/basic_grass.png";


constexpr r32 screen_height_m(r32 screen_width_m)
{
    return screen_width_m * app::SCREEN_BUFFER_HEIGHT / app::SCREEN_BUFFER_WIDTH;
}


constexpr r32 px_to_m(r32 n_pixels, r32 width_m, u32 width_px)
{
    return n_pixels * width_m / width_px;
}


constexpr size_t host_memory_sz()
{
    // HostImage screen_pixels;
    auto screen_pixels_data_sz = app::SCREEN_BUFFER_WIDTH * app::SCREEN_BUFFER_HEIGHT * sizeof(Pixel);

    return screen_pixels_data_sz;
}


bool init_device_memory(AppState& state);

bool init_unified_memory(AppState& state, app::ScreenBuffer& buffer);

void init_app_input(AppInput& app_input);