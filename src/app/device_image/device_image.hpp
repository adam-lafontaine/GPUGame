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

inline bool make_device_image(DeviceImage& image, device::MemoryBuffer& buffer, u32 width, u32 height)
{
    auto data_size = device_image_data_size(width, height);

    auto pixel_data = device::push_bytes(buffer, data_size);
    if(!pixel_data)
    {
        assert("make_device_image" && false);
        return false;
    }

    image.width = width;
    image.height = height;
    image.data = (Pixel*)pixel_data;

    return true;
}


inline bool copy_to_device(image_t const& src, DeviceImage const& dst)
{
    assert(src.data);
    assert(src.width);
    assert(src.height);
    assert(dst.data);
    assert(dst.width == src.width);
    assert(dst.height == src.height);

    auto bytes = src.width * src.height * sizeof(pixel_t);

    return cuda_memcpy_to_device(src.data, dst.data, bytes);
}


inline bool copy_to_host(DeviceImage const& src, image_t const& dst)
{
    assert(src.data);
    assert(src.width);
    assert(src.height);
    assert(dst.data);
    assert(dst.width == src.width);
    assert(dst.height == src.height);

    auto bytes = src.width * src.height * sizeof(pixel_t);

    return cuda_memcpy_to_host(src.data, dst.data, bytes);
}
