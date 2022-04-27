#pragma once

#include "../../device/device.hpp"


class DeviceTile
{
public:

    Pixel* bitmap_data;
    Pixel* avg_color;
};

bool copy_to_device(image_t const& src, DeviceTile const& dst);

namespace device
{
    bool push_device_tile(device::MemoryBuffer& buffer, DeviceTile& tile);
}