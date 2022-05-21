#include "device_image.hpp"


bool copy_to_device(image_t const& src, DeviceImage const& dst)
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


bool copy_to_host(DeviceImage const& src, image_t const& dst)
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


DeviceImage* make_device_image(device::MemoryBuffer& buffer, u32 width, u32 height)
{
    auto data_size = device_image_data_size(width, height);
    auto struct_size = sizeof(DeviceImage);

    auto pixel_data = device::push_bytes(buffer, data_size);
    if(!pixel_data)
    {
        return nullptr;
    }

    auto struct_data = device::push_bytes(buffer, struct_size);
    if(!struct_data)
    {
        device::pop_bytes(buffer, data_size);
        return nullptr;
    }

    DeviceImage image;
    image.width = width;
    image.height = height;
    image.data = (Pixel*)pixel_data;

    auto device_dst = (DeviceImage*)struct_data;

    if(!cuda_memcpy_to_device(&image, device_dst, struct_size))
    {
        device::pop_bytes(buffer, data_size);
        return nullptr;
    }

    return device_dst;
}


namespace device
{
    bool push_device_image(MemoryBuffer& buffer, DeviceImage& image, u32 width, u32 height)
    {
        auto data = push_bytes(buffer, width * height * sizeof(Pixel));

        if(data)
        {
            image.width = width;
            image.height = height;
            image.data = (pixel_t*)data;

            return true;
        }

        return false;
    }
}