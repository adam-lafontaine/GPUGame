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

    /***********************/

GPU_CONSTEXPR_FUNCTION
inline i32 floor_r32_to_i32(r32 value)
{
    return (i32)(floorf(value));
}


GPU_CONSTEXPR_FUNCTION
inline i32 ceil_r32_to_i32(r32 value)
{
    return (i32)(ceilf(value));
}


GPU_CONSTEXPR_FUNCTION
inline i32 round_r32_to_i32(r32 value)
{
    return (i32)(roundf(value));
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

    return (u32)gpu::ceil_r32_to_i32(px);
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
inline void update_position(WorldPosition& pos, Vec2Dr32 const& delta)
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
inline WorldPosition add_delta(WorldPosition const& pos, Vec2Dr32 const& delta)
{
    WorldPosition added = pos;

    gpu::update_position(added, delta);

    return added; 
}


GPU_FUNCTION
inline Vec2Di32 subtract(Vec2Di32 const& lhs, Vec2Di32 const& rhs)
{
    Vec2Di32 delta{};

    delta.x = lhs.x - rhs.x;
    delta.y = lhs.y - rhs.y;

    return delta;
}


GPU_FUNCTION
inline Vec2Dr32 sub_delta_m(WorldPosition const& lhs, WorldPosition const& rhs)
{
    Vec2Dr32 delta{};

    delta.x = TILE_LENGTH_M * (lhs.tile.x - rhs.tile.x) + lhs.offset_m.x - rhs.offset_m.x;
    delta.y = TILE_LENGTH_M * (lhs.tile.y - rhs.tile.y) + lhs.offset_m.y - rhs.offset_m.y;

    return delta;
}


GPU_FUNCTION
inline Rect2Dr32 make_rect(r32 width, r32 height)
{
    Rect2Dr32 r{};

    r.x_begin = 0.0f;
    r.x_end = width;
    r.y_begin = 0.0f;
    r.y_end = height;

    return r;
}


GPU_FUNCTION
inline Rect2Dr32 make_rect(Point2Dr32 const& begin, r32 width, r32 height)
{
    assert(width >= 0.0f);
    assert(height >= 0.0f);

    Rect2Dr32 r{};

    r.x_begin = begin.x;
    r.x_end = begin.x + width;
    r.y_begin = begin.y;
    r.y_end = begin.y + height;

    return r;
}


GPU_FUNCTION
inline Rect2Dr32 get_screen_rect(Entity const& entity, Point2Dr32 const& screen_pos)
{
    Rect2Dr32 r{};

    // pos at top left
    r.x_begin = screen_pos.x;
    r.x_end = r.x_begin + entity.width;
    r.y_begin = screen_pos.y;
    r.y_end = r.y_begin + entity.height;

    return r;
}


GPU_FUNCTION
inline bool rect_intersect(Rect2Dr32 const& a, Rect2Dr32 const& b)
{
    bool is_out = 
        a.x_end < b.x_begin ||
        b.x_end < a.x_begin ||
        a.y_end < b.y_begin ||
        b.y_end < a.y_begin;

    return !is_out;        
}


GPU_FUNCTION
inline void clamp_rect(Rect2Dr32& rect, Rect2Dr32 const& boundary)
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
inline Rect2Du32 to_pixel_rect(Rect2Dr32 const& rect_m, r32 length_m, u32 length_px)
{
    auto const to_px = [&](r32 m){ return gpu::m_to_px(m, length_m, length_px); };

    Rect2Du32 rect_px{};
    rect_px.x_begin = to_px(rect_m.x_begin);
    rect_px.x_end = to_px(rect_m.x_end);
    rect_px.y_begin = to_px(rect_m.y_begin);
    rect_px.y_end = to_px(rect_m.y_end);

    return rect_px;
}


GPU_FUNCTION
inline Vec2Dr32 vec_mul(Vec2Dr32 const& vec, r32 scale)
{
    Vec2Dr32 res{};
    res.x = vec.x * scale;
    res.y = vec.y * scale;

    return res;
}


GPU_CONSTEXPR_FUNCTION
inline bool is_player_entity(u32 id)
{
    return id == PLAYER_ID;
}


GPU_CONSTEXPR_FUNCTION
inline bool is_blue_entity(u32 id)
{
    auto begin = N_PLAYERS;
    auto end = begin + N_BLUE_ENTITIES;

    return id >= begin && id < end;
}


GPU_CONSTEXPR_FUNCTION
inline bool is_brown_entity(u32 id)
{
    auto begin = N_PLAYERS + N_BLUE_ENTITIES;
    auto end = begin + N_BROWN_ENTITIES;

    return id >= begin && id < end;
}


GPU_CONSTEXPR_FUNCTION
inline u32 get_blue_id(u32 id)
{
    return id - N_PLAYERS;
}


GPU_CONSTEXPR_FUNCTION
inline u32 get_brown_id(u32 id)
{
    return id - (N_PLAYERS + N_BLUE_ENTITIES);
}


GPU_CONSTEXPR_FUNCTION
inline u32 get_entity_id_from_brown_id(u32 id)
{
    return N_PLAYERS + N_BLUE_ENTITIES + id;
}


/***********************/

}


