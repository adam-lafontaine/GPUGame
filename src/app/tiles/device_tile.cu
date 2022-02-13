#include "device_tile.hpp"
#include "../gpu/gpu_constants.hpp"
#include "../../device/cuda_def.cuh"


static Pixel get_avg_color(image_t const& image)
{
    auto sub_h = image.height / 10;
    auto sub_w = image.width / 10;

    u32 r = 0;
    u32 g = 0;
    u32 b = 0;
    for(u32 y = 0; y < sub_h; ++y)
    {
        for(u32 x = 0; x < sub_w; ++x)
        {
            auto p = image.data[y * image.width + x];
            r += p.red;
            g += p.green;
            b += p.blue;
        }
    }

    auto div = sub_h * sub_w;
    r /= div;
    g /= div;
    b /= div;

    Pixel p{};
    p.alpha = 255;
    p.red = (u8)r;
    p.green = (u8)g;
    p.blue = (u8)b;

    return p;
}


bool make_device_tile(DeviceTile& tile, DeviceBuffer& buffer)
{
    assert(buffer.data);

    auto width = TILE_WIDTH_PX;
    auto height = TILE_HEIGHT_PX;

    auto bitmap_data_sz = width * height * sizeof(pixel_t);
    auto avg_color_sz = sizeof(pixel_t);

    auto bytes = bitmap_data_sz + avg_color_sz;

    bool result = buffer.total_bytes - buffer.offset >= bytes;
    if(result)
    {
        tile.bitmap_data = (pixel_t*)(buffer.data + buffer.offset);
        buffer.offset += bitmap_data_sz;

        tile.avg_color = (pixel_t*)(buffer.data + buffer.offset);
        buffer.offset += avg_color_sz;
    }

    return result;
}


bool copy_to_device(image_t const& src, DeviceTile const& dst)
{
    assert(src.data);
    assert(dst.bitmap_data);

    // image must first be resized
    assert(src.width == TILE_WIDTH_PX);
    assert(src.height == TILE_HEIGHT_PX);

    auto bytes = src.width * src.height * sizeof(pixel_t);

    auto result = cuda_memcpy_to_device(src.data, dst.bitmap_data, bytes);

    if(!result)
    {
        return false;
    }

    auto avg = get_avg_color(src);
    bytes = sizeof(pixel_t);

    return cuda_memcpy_to_device(&avg, dst.avg_color, bytes);    
}