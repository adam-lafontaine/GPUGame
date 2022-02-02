#include "render.hpp"
#include "../device/cuda_def.cuh"

#include <cassert>


constexpr int THREADS_PER_BLOCK = 1024;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


GPU_CONSTEXPR_FUNCTION r32 pixel_distance_m(u32 n_pixels, r32 width_m, u32 width_px)
{
    return n_pixels * width_m / width_px;
}


class DrawProps
{
public:

    WorldPosition screen_pos;
    u32 screen_width_px;
    r32 screen_width_m;
    DeviceMatrix tiles;
    DeviceImage screen_dst;

};



GPU_FUNCTION
static WorldPosition add_distance(WorldPosition const& pos, Vec2Dr32 const& vec)
{
    auto dist_y_m = pos.tile.y * TILE_LENGTH_M + pos.offset_m.y + vec.y;
    auto tile_y = (u32)(dist_y_m / TILE_LENGTH_M);
    auto offset_y = dist_y_m - tile_y * TILE_LENGTH_M;

    auto dist_x_m = pos.tile.x * TILE_LENGTH_M + pos.offset_m.x + vec.x;
    auto tile_x = (u32)(dist_x_m / TILE_LENGTH_M);
    auto offset_x = dist_x_m - tile_x * TILE_LENGTH_M;

    WorldPosition p{};
    p.tile.x = tile_x;
    p.tile.y = tile_y;
    p.offset_m.x = offset_x;
    p.offset_m.y = offset_y;

    return p;
}


GPU_KERNAL
static void gpu_render(DrawProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto pixel_id = (u32)t;

    auto pixel_y_px = pixel_id / props.screen_width_px;
    auto pixel_x_px = pixel_id - pixel_y_px * props.screen_width_px;

    auto pixel_y_m = pixel_distance_m(pixel_y_px, props.screen_width_m, props.screen_width_px);
    auto pixel_x_m = pixel_distance_m(pixel_x_px, props.screen_width_m, props.screen_width_px);

    auto pixel_pos = add_distance(props.screen_pos, { pixel_x_m, pixel_y_m });

    auto tile_id = pixel_pos.tile.y * WORLD_WIDTH_TILE + pixel_pos.tile.x;

    if(tile_id < WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE)
    {
        auto p = ((Pixel*)(props.tiles.data))[tile_id];
        props.screen_dst.data[pixel_id] = p;
    }
    else
    {
        Pixel p{};
        p.alpha = 255;
        p.red = 0;
        p.green = 0;
        p.blue = 0;

        props.screen_dst.data[pixel_id] = p;
    }
}


void render(AppState& state)
{
    auto& dst = state.unified.screen_pixels;

    u32 width = dst.width;
    u32 height = dst.height;
    u32 n_elements = width * height;

    auto n_threads = n_elements;

    DrawProps props{};
    props.screen_width_px = state.props.screen_width_px;
    props.screen_width_m = state.props.screen_width_m;
    props.screen_pos = state.props.screen_positon;
    props.tiles = state.device.tilemap;
    props.screen_dst = state.unified.screen_pixels;    

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_render<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


GPU_KERNAL
void gpu_init_tiles(DeviceMatrix tiles, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto tile_id = (u32)t;

    assert(tile_id < tiles.width * tiles.height);

    auto tile_y = tile_id / tiles.width;
    auto tile_x = tile_id - tile_y * tiles.width;

    auto p = (Pixel*)(tiles.data + tile_id);
    p->alpha = 255;

    if((tile_y % 2 == 0 && tile_x % 2 != 0) || (tile_y % 2 != 0 && tile_x % 2 == 0))
    {
        p->red = 255;
        p->green = 255;
        p->blue = 255;
    }
    else
    {
        p->red = (u8)(tile_y / 2);
        p->green = (u8)((tiles.width - tile_x) / 2);
        p->blue = (u8)((tiles.height - tile_y) / 2);
    }
}


void init_device_memory(DeviceMemory const& device)
{
    bool proc = cuda_no_errors();
    assert(proc);

    auto n_threads = device.tilemap.width * device.tilemap.height;
    gpu_init_tiles<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(device.tilemap, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}