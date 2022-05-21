#pragma once

#include "../../device/device.hpp"


constexpr u32 TILE_WIDTH_PX = 64;
constexpr u32 TILE_HEIGHT_PX = TILE_WIDTH_PX;
constexpr u32 N_TILE_BITMAPS = 3;


class DeviceTile
{
public:

    Pixel* bitmap_data;
    Pixel* avg_color;
};


constexpr size_t device_tile_bitmap_data_size()
{
    return TILE_WIDTH_PX * TILE_HEIGHT_PX * sizeof(Pixel);
}


constexpr size_t device_tile_avg_color_size()
{
    return sizeof(Pixel);
}


constexpr size_t device_tile_total_size()
{
    return 
        sizeof(DeviceTile) 
        + device_tile_bitmap_data_size()
        + device_tile_avg_color_size();
}


class TileList
{
public:
    DeviceTile grass;
    
    DeviceTile brown;
    DeviceTile black;
};


constexpr size_t tile_list_total_size()
{
    return sizeof(TileList) + N_TILE_BITMAPS * device_tile_total_size();
}


bool copy_to_device(image_t const& src, DeviceTile const& dst);

namespace device
{
    bool push_device_tile(device::MemoryBuffer& buffer, DeviceTile& tile);
}