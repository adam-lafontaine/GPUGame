#pragma once

#include "../utils/types.hpp"

#include <cstddef>
#include <cassert>
#include <array>


class ByteBuffer
{
public:
    u8* data = nullptr;
    size_t capacity = 0;
    size_t size = 0;
};


namespace cuda
{
    bool device_malloc(ByteBuffer& buffer, size_t n_bytes);

    bool unified_malloc(ByteBuffer& buffer, size_t n_bytes);

    bool free(ByteBuffer& buffer);


    u8* push_bytes(ByteBuffer& buffer, size_t n_bytes);

    bool pop_bytes(ByteBuffer& buffer, size_t n_bytes);


    bool memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes);

    bool memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes);


    bool no_errors(cstr label);

    bool launch_success(cstr label);
}


namespace device
{
    using u8 = uint8_t;

    class DeviceBuffer
    {
    public:
        u8* data = nullptr;
        size_t capacity = 0;
        size_t size = 0;
    };


    bool malloc(DeviceBuffer& buffer, size_t n_bytes);

    bool unified_malloc(DeviceBuffer& buffer, size_t n_bytes);

    bool free(DeviceBuffer& buffer);

    u8* push_bytes(DeviceBuffer& buffer, size_t n_bytes);

    bool pop_bytes(DeviceBuffer& buffer, size_t n_bytes);
}


template <typename T>
class DeviceArray
{
public:
    
    u32 n_elements = 0;
    T* data = nullptr;
};


template <class T, size_t N>
bool copy_to_device(std::array<T, N> const& src, DeviceArray<T>& dst)
{
    assert(dst.data);
    assert(dst.n_elements);
    assert(dst.n_elements == src.size());

    auto bytes = N * sizeof(T);

    return cuda::memcpy_to_device(src.data(), dst.data, bytes);
}


constexpr size_t device_image_data_size(u32 width, u32 height)
{
    return width * height * sizeof(Pixel);
}

inline bool make_device_image(Image& image, device::DeviceBuffer& buffer, u32 width, u32 height)
{
    auto data_size = device_image_data_size(width, height);

    auto pixel_data = device::push_bytes(buffer, data_size);
    if(!pixel_data)
    {
        return false;
    }

    image.width = width;
    image.height = height;
    image.data = (Pixel*)pixel_data;

    return true;
}