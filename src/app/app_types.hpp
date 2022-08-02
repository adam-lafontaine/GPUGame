#pragma once

#include "../device/device.hpp"
#include "../libimage/image_types.hpp"
#include "app_constants.hpp"




using uInput = u32;

namespace INPUT
{
    constexpr uInput PLAYER_UP    = 1;
    constexpr uInput PLAYER_DOWN  = PLAYER_UP * 2;
    constexpr uInput PLAYER_LEFT  = PLAYER_DOWN * 2;
    constexpr uInput PLAYER_RIGHT = PLAYER_LEFT * 2;

    constexpr u32 MAX_RECORDS = 5000;
}
 

class InputRecord
{
public:
    u64 frame_begin;
    u64 frame_end;
    uInput input;

    r32 est_dt_frame;
};


class InputList
{
public:
    u32 capacity;
    u32 size;
    u32 read_index;

    InputRecord* data;
};


template <u32 W, u32 H>
class Bitmap
{
public:
    Pixel* bitmap_data;
    Pixel* avg_color;

    static constexpr u32 width = W;
    static constexpr u32 height = H;
};



using Tile = Bitmap<TILE_WIDTH_PX, TILE_HEIGHT_PX>;
using PlayerBitmap = Bitmap<PLAYER_WIDTH_PX, PLAYER_HEIGHT_PX>;
using BlueBitmap = Bitmap<BLUE_WIDTH_PX, BLUE_HEIGHT_PX>;
using WallBitmap = Bitmap<WALL_WIDTH_PX, WALL_HEIGHT_PX>;


// bitmap + avg_color
constexpr auto N_TILE_BITMAP_PIXELS = TILE_WIDTH_PX * TILE_HEIGHT_PX + 1;
constexpr auto N_PLAYER_BITMAP_PIXELS = PLAYER_WIDTH_PX * PLAYER_HEIGHT_PX + 1;
constexpr auto N_BLUE_BITMAP_PIXELS = BLUE_WIDTH_PX * BLUE_HEIGHT_PX + 1;
constexpr auto N_WALL_BITMAP_PIXELS = WALL_WIDTH_PX * WALL_HEIGHT_PX + 1;


class DeviceAssets
{
public:
    Tile grass_tile;
    Tile black_tile;

    PlayerBitmap player_bitmap;
    BlueBitmap blue_bitmap;
    WallBitmap wall_bitmap;
};


constexpr size_t total_asset_pixel_size()
{
    u32 n_tiles = 2;
    u32 n_players = 1;
    u32 n_blue = 1;
    u32 n_wall = 1;    

    u32 n_pixels = 
        n_tiles * N_TILE_BITMAP_PIXELS +
        n_players * N_PLAYER_BITMAP_PIXELS +
        n_blue * N_BLUE_BITMAP_PIXELS +
        n_wall * N_WALL_BITMAP_PIXELS;

    return n_pixels * sizeof(Pixel);
}


class WorldPosition
{
public:
    Point2Di32 tile;
    Point2Dr32 offset_m;
};


using uStatus = u32;

namespace STATUS
{
    constexpr uStatus ACTIVE = 1;
    constexpr uStatus ONSCREEN = 2 * ACTIVE;
    constexpr uStatus INV_X = 2 * ONSCREEN;
    constexpr uStatus INV_Y = 2 * INV_X;
}


class Entity
{
public:
    u32 id;

    r32 width_m;
    r32 height_m;
    
    Image bitmap;
    Pixel avg_color;

    WorldPosition position;
    Vec2Dr32 dt;
    r32 speed;

    //b32 inv_x = false;
    //b32 inv_y = false;

    Vec2Dr32 delta_pos_m;

    WorldPosition next_position;

    uStatus status = 0;
};





using EntityArray = Array<Entity>;
using TileMatrix = Matrix<Tile>;


class AppInput
{
public:

    bool reset_frame_count;

    r32 screen_width_m;
    r32 screen_height_m;

    WorldPosition screen_position;
};


class DeviceMemory
{
public:

    DeviceAssets assets;

    TileMatrix tilemap;

    EntityArray entities;

    EntityArray player_entities;    
    EntityArray blue_entities;
    EntityArray wall_entities;

    Image screen_pixels;
};


class UnifiedMemory
{
public:

    u64 frame_count;    

    InputList previous_inputs;
    InputList current_inputs;

    u32 user_player_entity_id = 0;
};


class AppState
{
public:
    AppInput app_input;

    Image device_pixels;
    Image screen_pixels;

    MemoryBuffer<DeviceMemory> device_buffer;
    MemoryBuffer<Pixel> device_pixel_buffer;
    MemoryBuffer<Tile> device_tile_buffer;
    MemoryBuffer<Entity> device_entity_buffer;

    MemoryBuffer<UnifiedMemory> unified_buffer;
    MemoryBuffer<InputRecord> unified_input_record_buffer;
};