#include "device_tile.hpp"
#include "../gpu/gpu_app.hpp"


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

    return to_pixel((u8)r, (u8)g, (u8)b);
}


bool copy_to_device(image_t const& src, DeviceTile const& dst)
{
    assert(src.data);
    assert(dst.bitmap_data);

    // image must first be resized
    assert(src.width == TILE_WIDTH_PX);
    assert(src.height == TILE_HEIGHT_PX);

    auto bytes = src.width * src.height * sizeof(pixel_t);

    auto result = cuda::memcpy_to_device(src.data, dst.bitmap_data, bytes);

    if(!result)
    {
        return false;
    }

    auto avg = get_avg_color(src);
    bytes = sizeof(pixel_t);

    return cuda::memcpy_to_device(&avg, dst.avg_color, bytes);    
}