#pragma once

#include "../utils/types.hpp"

#include <cstddef>
#include <cassert>
#include <array>


bool cuda_memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes);

bool cuda_memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes);


bool cuda_no_errors();

bool cuda_launch_success();


class DeviceBuffer
{
public:
    u8* data = nullptr;
    u32 total_bytes = 0;
    u32 offset = 0;
};

bool device_malloc(DeviceBuffer& buffer, size_t n_bytes);

bool unified_malloc(DeviceBuffer& buffer, size_t n_bytes);

bool device_free(DeviceBuffer& buffer);


template <typename T>
class DeviceArray
{
public:
    T* data = nullptr;
    u32 n_elements = 0;
};


template <typename T>
bool make_device_array(DeviceArray<T>& arr, u32 n_elements, DeviceBuffer& buffer)
{
    assert(buffer.data);

    auto bytes = n_elements * sizeof(T);
    bool result = buffer.offset + bytes <= buffer.total_bytes;

    if(result)
    {
        arr.n_elements = n_elements;
        arr.data = (T*)((u8*)buffer.data + buffer.offset);
        buffer.offset += bytes;
    }

    return result;
}


template <class T, size_t N>
bool copy_to_device(std::array<T, N> const& src, DeviceArray<T>& dst)
{
    assert(dst.data);
    assert(dst.n_elements);
    assert(dst.n_elements == src.size());

    auto bytes = N * sizeof(T);

    return cuda_memcpy_to_device(src.data(), dst.data, bytes);
}


class DeviceImage
{
public:

    u32 width;
    u32 height;

    pixel_t* data;
};



bool make_device_image(DeviceImage& image, u32 width, u32 height, DeviceBuffer& buffer);

bool copy_to_device(image_t const& src, DeviceImage const& dst);

bool copy_to_host(DeviceImage const& src, image_t const& dst);


template <typename T>
class DeviceMatrix
{
public:
	u32 width;
	u32 height;

	T* data;
};


template <typename T>
bool make_device_matrix(DeviceMatrix<T>& matrix, u32 width, u32 height, DeviceBuffer& buffer)
{
    assert(buffer.data);

    auto n_elements = width * height;
    auto bytes = n_elements * sizeof(T);
    bool result = buffer.offset + bytes <= buffer.total_bytes;
    
    if(result)
    {
        matrix.width = width;
        matrix.height = height;
        matrix.data = (T*)((u8*)buffer.data + buffer.offset);
        buffer.offset += bytes;
    }

    return result;
}


class DeviceColorPalette
{
public:
    u8* channels[RGB_CHANNELS];

    u32 n_colors = 0;
};


bool make_device_palette(DeviceColorPalette& palette, u32 n_colors, DeviceBuffer& buffer);


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


using uInput = u8;

class InputRecord
{
public:
    u64 frame_begin;
    u64 frame_end;
    uInput input;

    r32 est_dt_frame;
};

constexpr uInput INPUT_PLAYER_UP    = 1;
constexpr uInput INPUT_PLAYER_DOWN  = 2;
constexpr uInput INPUT_PLAYER_LEFT  = 4;
constexpr uInput INPUT_PLAYER_RIGHT = 8;


class DeviceInputList
{
public:
    u32 capacity;
    u32 size;
    u32 read_index;

    InputRecord* data;
};