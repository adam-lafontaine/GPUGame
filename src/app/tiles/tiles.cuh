#pragma once

#include "../gpu/gpu_include.cuh"





/*
GPU_FUNCTION
void fill_image(image_t const& tile, Pixel p)
{
    assert(tile.data);
    assert(tile.width);
    assert(tile.height);

    for(u32 i = 0; i < tile.width * tile.height; ++i)
    {
        tile.data[i] = p;
    }
}
*/

/*
GPU_GLOBAL_VARIABLE
image_t GREEN_TILE;

GPU_GLOBAL_VARIABLE
image_t WHITE_TILE;
*/

GPU_FUNCTION
inline void create_tiles()
{
    
    
}


GPU_FUNCTION
inline Pixel green_tile()
{
    return to_pixel(40, 255, 40);
}


GPU_FUNCTION
inline Pixel white_tile()
{
    return to_pixel(255, 255, 255);
}