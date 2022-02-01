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

    u8 red;
    u8 green;
    u8 blue;
};



GPU_FUNCTION
static WorldPosition add_distance(WorldPosition const& pos, Vec2Dr32 const& vec)
{
    auto dist_y_m = pos.tile.y * TILE_LENGTH_M + pos.offset_m.y + vec.y;
    auto tile_y = (u32)(dist_y_m / TILE_LENGTH_M);
    auto offset_y = dist_y_m - tile_y - TILE_LENGTH_M;

    auto dist_x_m = pos.tile.x * TILE_LENGTH_M + pos.offset_m.x + vec.x;
    auto tile_x = (u32)(dist_x_m / TILE_LENGTH_M);
    auto offset_x = dist_x_m - tile_x - TILE_LENGTH_M;

    WorldPosition p{};
    p.tile.x = tile_x;
    p.tile.y = tile_y;
    p.offset_m.x = offset_x;
    p.offset_m.y = offset_y;

    return p;
}




GPU_FUNCTION
static void something(u32 pixel_id, u32 screen_width_px, WorldPosition const& screen_pos, r32 screen_width_m)
{
    auto pixel_y_px = pixel_id / screen_width_px;
    auto pixel_x_px = pixel_id - pixel_y_px * screen_width_px;

    auto pixel_y_m = pixel_distance_m(pixel_y_px, screen_width_m, screen_width_px);
    auto pixel_x_m = pixel_distance_m(pixel_x_px, screen_width_m, screen_width_px);
    
    auto pixel_pos = add_distance(screen_pos, { pixel_x_m, pixel_y_m});
}


GPU_FUNCTION
static void draw_pixel(Pixel* dst, u32 id)
{
    pixel_t p{};

    p.red = 128;
    p.green = 128;
    p.blue = 128;

    dst[id] = p;
}


GPU_KERNAL
static void gpu_render(Pixel* dst, DrawProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

   auto pixel_id = (u32)t;
   draw_pixel(dst, pixel_id);
}


void render(AppState& state)
{
    auto& dst = state.unified.screen_pixels;

    u32 width = dst.width;
    u32 height = dst.height;
    u32 n_elements = width * height;

    auto n_threads = n_elements;

    DrawProps props{};
    props.red = state.props.red;
    props.blue = state.props.blue;
    props.green = state.props.green;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_render<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(dst.data, props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}