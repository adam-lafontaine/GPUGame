#pragma once

#include "../../device/device.hpp"


class DeviceImage
{
public:

    u32 width;
    u32 height;

    Pixel* data;
};


constexpr size_t device_image_data_size(u32 width, u32 height)
{
    return width * height * sizeof(Pixel);
}


constexpr size_t device_image_total_size(u32 width, u32 height)
{
    return
        sizeof(DeviceImage)
        + device_image_data_size(width, height);
}


DeviceImage* make_device_image(device::MemoryBuffer& buffer, u32 width, u32 height);


bool copy_to_device(image_t const& src, DeviceImage const& dst);

bool copy_to_host(DeviceImage const& src, image_t const& dst);


namespace device
{
    bool push_device_image(MemoryBuffer& buffer, DeviceImage& image, u32 width, u32 height);

    
}