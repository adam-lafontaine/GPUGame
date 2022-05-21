#include "app.hpp"
#include "gpu/gpu_app.hpp"
#include "../libimage/libimage.hpp"

namespace img = libimage;

#include <cassert>
#include <cmath>

#define PRINT

#ifdef PRINT
#include <cstdio>
#endif

static void print(const char* msg)
{
#ifdef PRINT

    printf("* %s *\n", msg);

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


constexpr size_t device_memory_sz()
{
    // TileList tile_assets;
    auto tile_pixel_data_sz = TILE_HEIGHT_PX * TILE_WIDTH_PX * sizeof(Pixel);
    auto tile_avg_color_sz = sizeof(Pixel);
    auto device_tile_sz = tile_pixel_data_sz + tile_avg_color_sz;
    auto tile_assets_sz = N_TILE_BITMAPS * device_tile_sz;

    // DeviceTileMatrix tilemap;
    auto tilemap_data_sz = WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE * sizeof(DeviceTile);
    auto tilemap_sz = tilemap_data_sz;

    // DeviceArray<Entity> entities;
    auto entities_data_sz = N_ENTITIES * sizeof(Entity);
    auto entities_sz = entities_data_sz;

    // DeviceImage screen_pixels;
    auto const n_pixels = app::SCREEN_BUFFER_WIDTH * app::SCREEN_BUFFER_HEIGHT;
    auto screen_pixels_data_sz = sizeof(pixel_t) * n_pixels;
    auto screen_pixels_sz = screen_pixels_data_sz;

    // DeviceInputList* previous_inputs;
    auto previous_inputs_sz = device_input_list_total_size();
    
    return tile_assets_sz + tilemap_sz + entities_sz + screen_pixels_sz + previous_inputs_sz;
}


constexpr size_t unified_memory_sz()
{
    // DeviceInputList* current_inputs;
    auto const n_records = MAX_INPUT_RECORDS;
    auto current_inputs_data_sz = sizeof(InputRecord) * n_records;
    auto current_inputs_sz = current_inputs_data_sz + sizeof(DeviceInputList);

    return current_inputs_sz;
}