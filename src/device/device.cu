#include "device.hpp"
#include "cuda_def.cuh"

#ifdef CUDA_PRINT_ERROR

#include <cstdio>
#include <cstring>

#endif


static void check_error(cudaError_t err, cstr label = "")
{
    if(err == cudaSuccess)
    {
        return;
    }

    #ifdef CUDA_PRINT_ERROR

    printf("\n*** CUDA ERROR ***\n\n");
    printf("%s", cudaGetErrorString(err));

    if(std::strlen(label))
    {
        printf("\n%s", label);
    }
    
    printf("\n\n******************\n\n");

    #endif
}


bool cuda_memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes)
{
    cudaError_t err = cudaMemcpy(device_dst, host_src, n_bytes, cudaMemcpyHostToDevice);
    check_error(err, "cuda_memcpy_to_device");

    return err == cudaSuccess;
}


bool cuda_memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes)
{
    cudaError_t err = cudaMemcpy(host_dst, device_src, n_bytes, cudaMemcpyDeviceToHost);
    check_error(err, "cuda_memcpy_to_host");

    return err == cudaSuccess;
}


bool cuda_no_errors(cstr label)
{
    cudaError_t err = cudaGetLastError();
    check_error(err, label);

    return err == cudaSuccess;
}


bool cuda_launch_success(cstr label)
{
    cudaError_t err = cudaDeviceSynchronize();
    check_error(err, label);

    return err == cudaSuccess;
}





namespace device
{
    bool malloc(MemoryBuffer& buffer, size_t n_bytes)
    {
        assert(n_bytes);
        assert(!buffer.data);

        if(!n_bytes || buffer.data)
        {
            return false;
        }

        cudaError_t err = cudaMalloc((void**)&(buffer.data), n_bytes);
        check_error(err, "malloc");

        bool result = err == cudaSuccess;

        if(result)
        {
            buffer.capacity = n_bytes;
        }
        
        return result;
    }


    bool unified_malloc(MemoryBuffer& buffer, size_t n_bytes)
    {
        assert(n_bytes);
        assert(!buffer.data);

        if(!n_bytes || buffer.data)
        {
            return false;
        }

        cudaError_t err = cudaMallocManaged((void**)&(buffer.data), n_bytes);
        check_error(err, "unified_malloc");

        bool result = err == cudaSuccess;

        if(result)
        {
            buffer.capacity = n_bytes;
        }
        
        return result;
    }


    bool free(MemoryBuffer& buffer)
    {
        buffer.capacity = 0;
        buffer.size = 0;

        if(buffer.data)
        {
            cudaError_t err = cudaFree(buffer.data);
            check_error(err, "free");

            buffer.data = nullptr;

            return err == cudaSuccess;
        }

        return true;
    }


    u8* push_bytes(MemoryBuffer& buffer, size_t n_bytes)
    {
        assert(buffer.data);
        assert(buffer.capacity);
        assert(buffer.size < buffer.capacity);

        auto is_valid = 
            buffer.data &&
            buffer.capacity &&
            buffer.size < buffer.capacity;

        auto bytes_available = (buffer.capacity - buffer.size) >= n_bytes;
        assert(bytes_available);

        if(!is_valid || !bytes_available)
        {
            return nullptr;
        }

        auto data = buffer.data + buffer.size;

        buffer.size += n_bytes;

        return data;
    }


    bool pop_bytes(MemoryBuffer& buffer, size_t n_bytes)
    {
        assert(buffer.data);
        assert(buffer.capacity);
        assert(buffer.size <= buffer.capacity);
        assert(n_bytes <= buffer.capacity);
        assert(n_bytes <= buffer.size);

        auto is_valid = 
            buffer.data &&
            buffer.capacity &&
            buffer.size <= buffer.capacity &&
            n_bytes <= buffer.capacity &&
            n_bytes <= buffer.size;

        if(is_valid)
        {
            buffer.size -= n_bytes;
            return true;
        }

        return false;
    }


    


    bool push_device_palette(MemoryBuffer& buffer, DeviceColorPalette& palette, u32 n_colors)
    {
        auto bytes_per_channel = sizeof(u8) * n_colors;
        size_t bytes_allocated = 0;

        for(u32 c = 0; c < RGB_CHANNELS; ++c)
        {
            auto data = push_bytes(buffer, bytes_per_channel);
            if(!data)
            {
                break;                
            }

            bytes_allocated += bytes_per_channel;
            palette.channels[c] = (u8*)data;
        }

        if(bytes_allocated == RGB_CHANNELS * bytes_per_channel)
        {
            palette.n_colors = n_colors;
            return true;
        }
        else if (bytes_allocated > 0)
        {
            pop_bytes(buffer, bytes_allocated);            
        }

        return false;
    }
}