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


template<typename T>
class MemoryBuffer
{
public:
    T* data = nullptr;
    size_t capacity = 0;
    size_t size = 0;    
};


template<typename T>
ByteBuffer to_byte_buffer(MemoryBuffer<T> const& buffer)
{
    constexpr auto S = sizeof(T);

    ByteBuffer b{};
    b.data = (u8*)buffer.data;
    b.capacity = S * buffer.capacity;
    b.size = S * buffer.size;

    return b;
}


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


    template<typename T>
    bool device_malloc(MemoryBuffer<T>& buffer, size_t n_elements)
    {
        auto b = to_byte_buffer(buffer);
        
        if(!device_malloc(b, sizeof(T) * n_elements))
        {
            return false;
        }

        buffer.data = (T*)b.data;
        buffer.capacity = n_elements;
        buffer.size = 0;

        return true;
    }


    template<typename T>
    bool unified_malloc(MemoryBuffer<T>& buffer, size_t n_elements)
    {
        auto b = to_byte_buffer(buffer);
        
        if(!unified_malloc(b, sizeof(T) * n_elements))
        {
            return false;
        }

        buffer.data = (T*)b.data;
        buffer.capacity = n_elements;
        buffer.size = 0;

        return true;
    }


    template<typename T>
    bool free(MemoryBuffer<T>& buffer)
    {
        buffer.capacity = 0;
        buffer.size = 0;

        auto b = to_byte_buffer(buffer);

        return cuda::free(b);
    }


    template<typename T>
    T* push_elements(MemoryBuffer<T>& buffer, size_t n_elements)
    {
        auto b = to_byte_buffer(buffer);

        auto data = push_bytes(b, sizeof(T) * n_elements);

        if(!data)
        {
            return nullptr;
        }
        
        buffer.size += n_elements;

        return (T*)data;
        
    }


    template<typename T>
    bool pop_elements(MemoryBuffer<T>& buffer, size_t n_elements)
    {
        auto b = to_byte_buffer(buffer);

        if(!pop_bytes(b, sizeof(T) * n_elements))
        {
            return false;
        }
        
        buffer.size -= n_elements;

        return true;
    }
}