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


template <typename T>
class DeviceQueue
{
public:
    u32 capacity;
    u32 size;

    u32 index;

    T* data;    
};


template <typename T>
inline bool make_device_queue(DeviceQueue<T>& queue, u32 n_elements, DeviceBuffer& buffer)
{
    assert(buffer.data);

    auto bytes = n_elements * sizeof(T);
    bool result = buffer.offset + bytes <= buffer.total_bytes;

    if(result)
    {
        queue.capacity = n_elements;
        queue.size = 0;
        queue.index = 0;
        queue.data = (T*)((u8*)buffer.data + buffer.offset);
        buffer.offset += bytes;
    }

    return result;
}


template <typename T>
inline void push_back(DeviceQueue<T>& queue, T& item)
{
    assert(queue.data);
    assert(queue.capacity);
    assert(queue.size <= queue.capacity);

    queue.data[queue.size++] = item;
}


template <typename T>
inline T& get_next(DeviceQueue<T>& queue)
{
    assert(queue.data);
    assert(queue.index < queue.size);

    return queue.data[queue.index++];
}