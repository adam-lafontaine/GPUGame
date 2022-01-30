#include "render.hpp"
#include "../device/cuda_def.cuh"
#include <cassert>


constexpr int THREADS_PER_BLOCK = 1024;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


class DrawProps
{
public:

    u8 red;
    u8 green;
    u8 blue;
};


GPU_KERNAL
static void gpu_draw(Pixel* dst, DrawProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    pixel_t p{};
    p.red = props.red;
    p.green = props.green;
    p.blue = props.blue;

    dst[t] = p;
}


static void draw(DeviceImage const& dst, AppState const& state)
{
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

    gpu_draw<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(dst.data, props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


void render(AppState& state)
{
    draw(state.unified.screen_pixels, state);
}