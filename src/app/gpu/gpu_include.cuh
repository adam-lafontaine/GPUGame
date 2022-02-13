#pragma once

#include "gpu_app.hpp"
#include "../../device/cuda_def.cuh"

#include <cassert>


constexpr int THREADS_PER_BLOCK = 1024;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


namespace gpu
{

GPU_CONSTEXPR_FUNCTION
inline i32 floor_r32_to_i32(r32 value)
{
    return (i32)(floorf(value));
}


GPU_CONSTEXPR_FUNCTION
inline i32 round_r32_to_u32(r32 value)
{
    return (u32)(value + 0.5f);
}


GPU_CONSTEXPR_FUNCTION
inline r32 px_to_m(u32 n_pixels, r32 length_m, u32 length_px)
{
    auto dist = n_pixels * length_m / length_px;

    if(fabs(dist) < 0.0001)
    {
        return 0.0f;
    }

    return dist;
}


GPU_CONSTEXPR_FUNCTION
inline u32 m_to_px(r32 dist_m, r32 length_m, u32 length_px)
{
    auto px = dist_m * length_px / length_m;

    // never have negative pixels
    if(px < 0.0f)
    {
        px = 0.0f;
    }

    return gpu::round_r32_to_u32(px);
}


GPU_CONSTEXPR_FUNCTION
inline Pixel to_pixel(u8 red, u8 green, u8 blue)
{
    Pixel p{};
    p.alpha = 255;
    p.red = red;
    p.green = green;
    p.blue = blue;

    return p;
}


GPU_FUNCTION
static void update_position(WorldPosition& pos, Vec2Dr32 const& delta)
{
    r32 dist_m = pos.offset_m.x + delta.x;
    i32 delta_tile = gpu::floor_r32_to_i32(dist_m / TILE_LENGTH_M);
    
    pos.tile.x = pos.tile.x + delta_tile;
    pos.offset_m.x = dist_m - (r32)delta_tile * TILE_LENGTH_M;

    dist_m = pos.offset_m.y + delta.y;
    delta_tile = gpu::floor_r32_to_i32(dist_m / TILE_LENGTH_M);
    
    pos.tile.y = pos.tile.y + delta_tile;
    pos.offset_m.y = dist_m - (r32)delta_tile * TILE_LENGTH_M;
}


GPU_FUNCTION
static WorldPosition add_delta(WorldPosition const& pos, Vec2Dr32 const& delta)
{
    WorldPosition added{};

    r32 dist_m = pos.offset_m.x + delta.x;
    i32 delta_tile = gpu::floor_r32_to_i32(dist_m / TILE_LENGTH_M);
    
    added.tile.x = pos.tile.x + delta_tile;
    added.offset_m.x = dist_m - (r32)delta_tile * TILE_LENGTH_M;

    dist_m = pos.offset_m.y + delta.y;
    delta_tile = gpu::floor_r32_to_i32(dist_m / TILE_LENGTH_M);
    
    added.tile.y = pos.tile.y + delta_tile;
    added.offset_m.y = dist_m - (r32)delta_tile * TILE_LENGTH_M;

    return added; 
}


GPU_FUNCTION
static Vec2Dr32 subtract(WorldPosition const& lhs, WorldPosition const& rhs)
{
    Vec2Dr32 delta{};

    delta.x = TILE_LENGTH_M * (lhs.tile.x - rhs.tile.x) + lhs.offset_m.x - rhs.offset_m.x;
    delta.y = TILE_LENGTH_M * (lhs.tile.y - rhs.tile.y) + lhs.offset_m.y - rhs.offset_m.y;

    return delta;
}


GPU_FUNCTION
static Rect2Dr32 make_rect(r32 width, r32 height)
{
    Rect2Dr32 r{};

    r.x_begin = 0.0f;
    r.x_end = width;
    r.y_begin = 0.0f;
    r.y_end = height;

    return r;
}


GPU_FUNCTION
static Rect2Dr32 get_entity_rect(Entity const& entity, Point2Dr32 const& pos)
{
    Rect2Dr32 r{};

    // pos at bottom center of rect
    r.x_begin = pos.x - 0.5f * entity.width;
    r.x_end = r.x_begin + entity.width;
    r.y_end = pos.y;
    r.y_begin = r.y_end - entity.height;

    return r;
}


GPU_FUNCTION
static bool rect_intersect(Rect2Dr32 const& a, Rect2Dr32 const& b)
{
    bool is_out = 
        a.x_end < b.x_begin ||
        b.x_end < a.x_begin ||
        a.y_end < b.y_begin ||
        b.y_end < a.y_begin;

    return !is_out;        
}


GPU_FUNCTION
static void clamp_rect(Rect2Dr32& rect, Rect2Dr32 const& boundary)
{
    if(rect.x_begin < boundary.x_begin)
    {
        rect.x_begin = boundary.x_begin;
    }

    if(rect.x_end > boundary.x_end)
    {
        rect.x_end = boundary.x_end;
    }

    if(rect.y_begin < boundary.y_begin)
    {
        rect.y_begin = boundary.y_begin;
    }

    if(rect.y_end > boundary.y_end)
    {
        rect.y_end = boundary.y_end;
    }
}


GPU_FUNCTION
static Rect2Du32 to_pixel_rect(Rect2Dr32 const& rect_m, r32 length_m, u32 length_px)
{
    auto const to_px = [&](r32 m){ return gpu::m_to_px(m, length_m, length_px); };

    Rect2Du32 rect_px{};
    rect_px.x_begin = to_px(rect_m.x_begin);
    rect_px.x_end = to_px(rect_m.x_end);
    rect_px.y_begin = to_px(rect_m.y_begin);
    rect_px.y_end = to_px(rect_m.y_end);

    return rect_px;
}

}


