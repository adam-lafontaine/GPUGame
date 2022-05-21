#pragma once

#include "../utils/types.hpp"

#include <cstddef>
#include <cassert>
#include <array>


bool cuda_memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes);

bool cuda_memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes);


bool cuda_no_errors(cstr label);

bool cuda_launch_success(cstr label);


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

    return cuda_memcpy_to_device(src.data(), dst.data, bytes);
}



template <typename T>
class DeviceMatrix
{
public:
	u32 width;
	u32 height;

	T* data;
};


class DeviceColorPalette
{
public:
    u8* channels[RGB_CHANNELS];

    u32 n_colors = 0;
};


template <size_t N>
bool copy_to_device(std::array< std::array<u8, N>, RGB_CHANNELS> const& src, DeviceColorPalette& dst)
{
    assert(dst.channels[0]);
    assert(dst.n_colors);
    assert(dst.n_colors == src[0].size());

    auto bytes = src[0].size() * sizeof(u8);
    for(u32 c = 0; c < RGB_CHANNELS; ++c)
    {
        if(!cuda_memcpy_to_device(src[c].data(), dst.channels[c], bytes))
        {
            return false;
        }
    }

    return true;
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

    bool push_device_palette(MemoryBuffer& buffer, DeviceColorPalette& palette, u32 n_colors);


    template <typename T>
    inline bool push_device_array(MemoryBuffer& buffer, DeviceArray<T>& arr, u32 n_elements)
    {
        auto data = push_bytes(buffer, n_elements * sizeof(T));

        if(data)
        {
            arr.n_elements = n_elements;
            arr.data = (T*)data;

            return true;
        }

        return false;
    }


    template <typename T>
    inline bool push_device_matrix(MemoryBuffer& buffer, DeviceMatrix<T>& matrix, u32 width, u32 height)
    {
        auto n_bytes = sizeof(T) * width * height;
        auto data = push_bytes(buffer, n_bytes);

        if(data)
        {
            matrix.width = width;
            matrix.height = height;
            matrix.data = (T*)data;

            return true;
        }

        return false;
    }

}