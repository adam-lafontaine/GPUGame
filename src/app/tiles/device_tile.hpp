#pragma once

#include "../../device/device.hpp"


constexpr u32 TILE_WIDTH_PX = 64;
constexpr u32 TILE_HEIGHT_PX = TILE_WIDTH_PX;


class DeviceTile
{
public:

    Pixel* bitmap_data;
    Pixel* avg_color;
};

// bitmap + avg_color
constexpr auto N_TILE_PIXELS = TILE_WIDTH_PX * TILE_HEIGHT_PX + 1;


constexpr size_t device_tile_bitmap_data_size()
{
    return TILE_WIDTH_PX * TILE_HEIGHT_PX * sizeof(Pixel);
}


constexpr size_t device_tile_avg_color_size()
{
    return sizeof(Pixel);
}


constexpr size_t device_tile_data_size()
{
    return 
        device_tile_avg_color_size()
        + device_tile_bitmap_data_size();
}


inline bool make_device_tile(DeviceTile& tile, device::DeviceBuffer& buffer)
{    
    auto bitmap_data = device::push_bytes(buffer, device_tile_bitmap_data_size());
    if(!bitmap_data)
    {
        assert("make_device_tile: bitmap_data" && false);
        return false;
    }

    auto avg_color_data = device::push_bytes(buffer, device_tile_avg_color_size());
    if(!avg_color_data)
    {
        assert("make_device_tile: avg_color" && false);
        return false;
    }

    tile.bitmap_data = (Pixel*)bitmap_data;
    tile.avg_color = (Pixel*)avg_color_data;

    return true;
}


bool copy_to_device(Image const& src, DeviceTile const& dst);