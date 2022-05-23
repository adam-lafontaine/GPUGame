#pragma once

#include "../utils/types.hpp"

#include <cstddef>
#include <cassert>
#include <array>


namespace cuda
{
    bool memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes);

    bool memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes);


    bool no_errors(cstr label);

    bool launch_success(cstr label);
}


namespace device
{
    using u8 = uint8_t;

    class MemoryBuffer
    {
    public:
        u8* data = nullptr;
        size_t capacity = 0;
        size_t size = 0;
    };


    bool malloc(MemoryBuffer& buffer, size_t n_bytes);

    bool unified_malloc(MemoryBuffer& buffer, size_t n_bytes);

    bool free(MemoryBuffer& buffer);

    u8* push_bytes(MemoryBuffer& buffer, size_t n_bytes);

    bool pop_bytes(MemoryBuffer& buffer, size_t n_bytes);
}


template <typename T>
class DeviceArray
{
public:
    T* data = nullptr;
    u32 n_elements = 0;
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


template <typename T>
class DeviceMatrix
{
public:
	u32 width;
	u32 height;

	T* data = nullptr;
};


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

    return cuda::memcpy_to_device(src.data, dst.data, bytes);
}

/*
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
*/