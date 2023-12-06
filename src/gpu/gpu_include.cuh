#pragma once

#include "gpu_app.hpp"
#include "../device/cuda_def.cuh"

class ScreenProps
{
public:
    DeviceMemory* device_p;

    WorldPosition screen_pos;
    f32 screen_width_m;
    f32 screen_height_m;
};


inline ScreenProps make_screen_props(AppState const& state)
{
    ScreenProps props{};
    props.device_p = state.device_buffer.data;
    props.screen_width_m = state.app_input.screen_width_m;
    props.screen_height_m = state.app_input.screen_height_m;
    props.screen_pos = state.app_input.screen_position;

    return props;
}


class PlayerProps
{
public:
    PlayerEntitySOA props;
    u32 id;
};


class BlueProps
{
public:
    BlueEntitySOA props;
    u32 id;
};


class WallProps
{
public:
    WallEntitySOA props;
    u32 id;
};


constexpr int THREADS_PER_BLOCK = 512;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


namespace gpuf
{

/***********************/

GPU_CONSTEXPR_FUNCTION
inline i32 floor_f32_to_i32(f32 value)
{
    return (i32)(floorf(value));
}


GPU_CONSTEXPR_FUNCTION
inline i32 ceil_f32_to_i32(f32 value)
{
    return (i32)(ceilf(value));
}


GPU_CONSTEXPR_FUNCTION
inline i32 round_f32_to_i32(f32 value)
{
    return (i32)(roundf(value));
}


GPU_CONSTEXPR_FUNCTION
inline f32 px_to_m(u32 n_pixels, f32 length_m, u32 length_px)
{
    auto dist = n_pixels * length_m / length_px;

    return dist;
}


GPU_CONSTEXPR_FUNCTION
inline i32 m_to_px(f32 dist_m, f32 length_m, u32 length_px)
{
    auto px = dist_m * length_px / length_m;

    return gpuf::floor_f32_to_i32(px);
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
inline void update_position(WorldPosition& pos, Vec2Df32 const& delta)
{
    f32 dist_m = pos.offset_m.x + delta.x;
    i32 delta_tile = gpuf::floor_f32_to_i32(dist_m / TILE_LENGTH_M);
    
    pos.tile.x = pos.tile.x + delta_tile;
    pos.offset_m.x = dist_m - (f32)delta_tile * TILE_LENGTH_M;

    dist_m = pos.offset_m.y + delta.y;
    delta_tile = gpuf::floor_f32_to_i32(dist_m / TILE_LENGTH_M);
    
    pos.tile.y = pos.tile.y + delta_tile;
    pos.offset_m.y = dist_m - (f32)delta_tile * TILE_LENGTH_M;
}


GPU_FUNCTION
inline WorldPosition add_delta(WorldPosition const& pos, Vec2Df32 const& delta)
{
    WorldPosition added = pos;

    gpuf::update_position(added, delta);

    return added; 
}


GPU_FUNCTION
inline Vec2Df32 add(Vec2Df32 const& lhs, Vec2Df32 const& rhs)
{
    Vec2Df32 vec{};

    vec.x = lhs.x + rhs.x;
    vec.y = lhs.y + rhs.y;

    return vec;
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
inline Vec2Df32 subtract(Vec2Df32 const& lhs, Vec2Df32 const& rhs)
{
    Vec2Df32 delta{};

    delta.x = lhs.x - rhs.x;
    delta.y = lhs.y - rhs.y;

    return delta;
}


GPU_FUNCTION
inline bool equal(Vec2Df32 const& lhs, Vec2Df32 const& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}


GPU_FUNCTION
inline Vec2Df32 sub_delta_m(WorldPosition const& pos, WorldPosition const& origin)
{
    Vec2Df32 delta{};

    delta.x = TILE_LENGTH_M * (pos.tile.x - origin.tile.x) + pos.offset_m.x - origin.offset_m.x;
    delta.y = TILE_LENGTH_M * (pos.tile.y - origin.tile.y) + pos.offset_m.y - origin.offset_m.y;

    return delta;
}


GPU_FUNCTION
inline Point2Df32 subtract_abs(WorldPosition const& pos, WorldPosition const& origin)
{
    Point2Df32 delta{};

    delta.x = TILE_LENGTH_M * (pos.tile.x - origin.tile.x) + pos.offset_m.x - origin.offset_m.x;
    delta.y = TILE_LENGTH_M * (pos.tile.y - origin.tile.y) + pos.offset_m.y - origin.offset_m.y;

    return delta;
}


GPU_FUNCTION
inline WorldPosition subtract(WorldPosition const& pos, WorldPosition const& origin)
{
    WorldPosition delta{};

    delta.tile.x = pos.tile.x - origin.tile.x;
    delta.tile.y = pos.tile.y - origin.tile.y;

    delta.offset_m.x = pos.offset_m.x - origin.offset_m.x;
    delta.offset_m.y = pos.offset_m.y - origin.offset_m.y;

    return delta;
}


GPU_FUNCTION
inline Rect2Df32 make_rect(f32 width, f32 height)
{
    Rect2Df32 r{};

    r.x_begin = 0.0f;
    r.x_end = width;
    r.y_begin = 0.0f;
    r.y_end = height;

    return r;
}


GPU_FUNCTION
inline Rect2Df32 make_rect(Point2Df32 const& begin, f32 width, f32 height)
{
    assert(width > 0.0f);
    assert(height > 0.0f);

    Rect2Df32 r{};

    r.x_begin = begin.x;
    r.x_end = begin.x + width;
    r.y_begin = begin.y;
    r.y_end = begin.y + height;

    return r;
}


GPU_FUNCTION
inline Rect2Df32 add_delta(Rect2Df32 const& rect, Vec2Df32 const& delta)
{
    Rect2Df32 r = rect;

    r.x_begin += delta.x;
    r.x_end += delta.x;
    r.y_begin += delta.y;
    r.y_end += delta.y;

    return r;
}


GPU_FUNCTION
inline bool rect_intersect(Rect2Df32 const& a, Rect2Df32 const& b)
{
    bool is_out = 
        a.x_end < b.x_begin ||
        b.x_end < a.x_begin ||
        a.y_end < b.y_begin ||
        b.y_end < a.y_begin;

    return !is_out;        
}


GPU_FUNCTION
inline void clamp_rect(Rect2Df32& rect, Rect2Df32 const& boundary)
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
inline Rect2Di32 to_pixel_rect(Rect2Df32 const& rect_m, f32 length_m, u32 length_px)
{
    auto const m_px = length_px / length_m;

    Rect2Di32 rect_px{};
    rect_px.x_begin = gpuf::floor_f32_to_i32(rect_m.x_begin * m_px);
    rect_px.x_end = gpuf::ceil_f32_to_i32(rect_m.x_end * m_px);
    rect_px.y_begin = gpuf::floor_f32_to_i32(rect_m.y_begin * m_px);
    rect_px.y_end = gpuf::ceil_f32_to_i32(rect_m.y_end * m_px);

    return rect_px;
}


GPU_FUNCTION
inline Vec2Df32 vec_mul(Vec2Df32 const& vec, f32 scale)
{
    Vec2Df32 res{};
    res.x = vec.x * scale;
    res.y = vec.y * scale;

    return res;
}


GPU_FUNCTION
inline bool id_in_range(u32 id, u32 begin, u32 end)
{
    if(begin == 0)
    {
        return id < end;
    }

    return begin <= id && id < end;
}


GPU_FUNCTION
inline u32 to_entity_id_from_player(u32 player_offset)
{
    return PLAYER_BEGIN + player_offset;
}


GPU_FUNCTION
inline u32 to_entity_id_from_blue(u32 blue_offset)
{
    return BLUE_BEGIN + blue_offset;
}


GPU_FUNCTION
inline u32 to_entity_id_from_wall(u32 wall_offset)
{
    return WALL_BEGIN + wall_offset;
}


GPU_FUNCTION
inline bool is_player(u32 entity_id)
{
    return gpuf::id_in_range(entity_id, PLAYER_BEGIN, PLAYER_END);
}


GPU_FUNCTION
inline bool is_blue(u32 entity_id)
{
    return gpuf::id_in_range(entity_id, BLUE_BEGIN, BLUE_END);
}


GPU_FUNCTION
inline bool is_wall(u32 entity_id)
{
    return gpuf::id_in_range(entity_id, WALL_BEGIN, WALL_END);
}


GPU_FUNCTION
inline u32 to_player_id(u32 entity_id)
{
    assert(gpuf::is_player(entity_id));

    return entity_id - PLAYER_BEGIN;
}


GPU_FUNCTION
inline u32 to_blue_id(u32 entity_id)
{
    assert(gpuf::is_blue(entity_id));

    return entity_id - BLUE_BEGIN;
}


GPU_FUNCTION
inline u32 to_wall_id(u32 entity_id)
{
    assert(gpuf::is_wall(entity_id));

    return entity_id - WALL_BEGIN;
}


GPU_FUNCTION
inline bool is_active(uStatus status)
{
    return status & STATUS::ACTIVE;
}


GPU_FUNCTION
inline void set_active(uStatus& status)
{
    status |= STATUS::ACTIVE;
}


GPU_FUNCTION
inline void set_inactive(uStatus& status)
{
    status &= ~STATUS::ACTIVE;
}


GPU_FUNCTION
inline bool is_onscreen(uStatus status)
{
    return status & STATUS::ONSCREEN;
}


GPU_FUNCTION
inline void set_onscreen(uStatus& status)
{
    status |= STATUS::ONSCREEN;
}


GPU_FUNCTION
inline void set_offscreen(uStatus& status)
{
    status &= ~STATUS::ONSCREEN;
}


GPU_FUNCTION
inline bool is_inv_x(uStatus status)
{
    return status & STATUS::INV_X;
}


GPU_FUNCTION
inline void set_inv_x(uStatus& status)
{
    status |= STATUS::INV_X;
}


GPU_FUNCTION
inline void reset_inv_x(uStatus& status)
{
    status &= ~STATUS::INV_X;
}


GPU_FUNCTION
inline bool is_inv_y(uStatus status)
{
    return status & STATUS::INV_Y;
}


GPU_FUNCTION
inline void set_inv_y(uStatus& status)
{
    status |= STATUS::INV_Y;
}


GPU_FUNCTION
inline void reset_inv_y(uStatus& status)
{
    status &= ~STATUS::INV_Y;
}


GPU_FUNCTION
inline bool is_stop_x(uStatus status)
{
    return status & STATUS::STOP_X;
}


GPU_FUNCTION
inline void set_stop_x(uStatus& status)
{
    status |= STATUS::STOP_X;
}


GPU_FUNCTION
inline void reset_stop_x(uStatus& status)
{
    status &= ~STATUS::STOP_X;
}


GPU_FUNCTION
inline bool is_stop_y(uStatus status)
{
    return status & STATUS::STOP_Y;
}


GPU_FUNCTION
inline void set_stop_y(uStatus& status)
{
    status |= STATUS::STOP_Y;
}


GPU_FUNCTION
inline void reset_stop_y(uStatus& status)
{
    status &= ~STATUS::STOP_Y;
}


GPU_FUNCTION
inline bool is_drawable(uStatus status)
{
    return is_active(status) && is_onscreen(status);
}

/***********************/

}


