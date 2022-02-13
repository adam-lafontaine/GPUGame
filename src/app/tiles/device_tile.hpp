#pragma once

#include "../../device/device.hpp"


class DeviceTile
{
public:

    Pixel* bitmap_data;
    Pixel* avg_color;
};


bool make_device_tile(DeviceTile& tile, DeviceBuffer& buffer);

bool copy_to_device(image_t const& src, DeviceTile const& dst);