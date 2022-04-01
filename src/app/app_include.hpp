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


constexpr size_t device_memory_sz()
{
    u32 n_world_tiles = WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE;  
    auto tilemap_sz = n_world_tiles * sizeof(DeviceTile);

    auto entity_sz = N_ENTITIES * sizeof(Entity);

    auto tile_asset_sz = N_TILE_BITMAPS * (TILE_HEIGHT_PX * TILE_WIDTH_PX * sizeof(Pixel) + sizeof(Pixel));

    auto const n_records = MAX_INPUT_RECORDS;
    auto input_record_sz = sizeof(InputRecord) * n_records + sizeof(DeviceInputList);

    return tilemap_sz + entity_sz + tile_asset_sz + input_record_sz;
}


constexpr size_t unified_memory_sz(u32 screen_width_px, u32 screen_height_px)
{
    auto const n_pixels = screen_width_px * screen_height_px;
    auto screen_sz = sizeof(pixel_t) * n_pixels;

    auto const n_records = MAX_INPUT_RECORDS;
    auto input_record_sz = sizeof(InputRecord) * n_records + sizeof(DeviceInputList);

    return screen_sz + input_record_sz;
}