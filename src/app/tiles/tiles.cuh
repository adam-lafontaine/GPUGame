#pragma once

#include "../gpu/gpu_include.cuh"


namespace tiles
{
    GPU_GLOBAL_VARIABLE Pixel* GREEN_TILE_DATA;

    GPU_GLOBAL_VARIABLE Pixel* WHITE_TILE_DATA; 
}





GPU_FUNCTION
inline void create_tiles(DeviceArray<Pixel> const& bitmap_data)
{
    auto pixel_data = bitmap_data.data;
    u32 bitmap_sz = TILE_BITMAP_LENGTH * TILE_BITMAP_LENGTH;

    tiles::GREEN_TILE_DATA = pixel_data;
    pixel_data += bitmap_sz;

    for(u32 i = 0; i < bitmap_sz; ++i)
    {
        tiles::GREEN_TILE_DATA[i] = to_pixel(90, 255, 20);
    }

    tiles::WHITE_TILE_DATA = pixel_data;
    pixel_data += bitmap_sz;

    for(u32 i = 0; i < bitmap_sz; ++i)
    {
        tiles::WHITE_TILE_DATA[i] = to_pixel(255, 255, 255);
    }
}


GPU_FUNCTION
inline Pixel green_tile()
{
    return tiles::GREEN_TILE_DATA[0];
}


GPU_FUNCTION
inline Pixel white_tile()
{
    return tiles::WHITE_TILE_DATA[0];
}