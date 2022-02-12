#pragma once

#include "../gpu/gpu_include.cuh"


namespace tiles
{
    GPU_GLOBAL_VARIABLE Tile GREEN_TILE;

    GPU_GLOBAL_VARIABLE Tile WHITE_TILE;
}





GPU_FUNCTION
inline void create_tiles(DeviceArray<Pixel> const& bitmap_data)
{
    auto pixel_data = bitmap_data.data;
    u32 bitmap_sz = TILE_BITMAP_LENGTH * TILE_BITMAP_LENGTH;

    tiles::GREEN_TILE.bitmap_data = pixel_data;
    pixel_data += bitmap_sz;

    auto green = to_pixel(90, 255, 80);
    auto black = to_pixel(0, 0, 0);
    auto white = to_pixel(255, 255, 255);

    u32 j = 0;

    for(u32 y = 0; y < TILE_BITMAP_LENGTH; ++y)
    {
        for(u32 x = 0; x < TILE_BITMAP_LENGTH; ++x, ++j)
        {
            if(y % 8 == 0 || (y + 1) % 8 == 0 || x % 8 == 0 || (x + 1) % 8 == 0)
            {
                tiles::GREEN_TILE.bitmap_data[j] = black;
            }
            else
            {
                tiles::GREEN_TILE.bitmap_data[j] = green;
            }
        }
    }

    auto sub_dim = TILE_BITMAP_LENGTH / 10;
    u32 r = 0;
    u32 g = 0;
    u32 b = 0;
    for(u32 y = 0; y < sub_dim; ++y)
    {
        for(u32 x = 0; x < sub_dim; ++x)
        {
            auto p = tiles::GREEN_TILE.bitmap_data[y * TILE_BITMAP_LENGTH + x];
            r += p.red;
            g += p.green;
            b += p.blue;
        }
    }   

    auto div = sub_dim * sub_dim;
    r /= div;
    g /= div;
    b /= div;
    tiles::GREEN_TILE.avg_color = to_pixel((u8)r, (u8)g, (u8)b);

    tiles::WHITE_TILE.bitmap_data = pixel_data;
    pixel_data += bitmap_sz;

    for(u32 i = 0; i < bitmap_sz; ++i)
    {
        tiles::WHITE_TILE.bitmap_data[i] = white;
    }

    tiles::WHITE_TILE.avg_color = white;
}