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





class DeviceAssets
{
public:
    Tile grass_tile;
    Tile black_tile;

    PlayerBitmap player_bitmap;
    BlueBitmap blue_bitmap;
    WallBitmap wall_bitmap;
};


namespace SIZE
{
    // bitmap + avg_color
    constexpr auto N_TILE_BITMAP_PIXELS = TILE_WIDTH_PX * TILE_HEIGHT_PX + 1;
    constexpr auto N_PLAYER_BITMAP_PIXELS = PLAYER_WIDTH_PX * PLAYER_HEIGHT_PX + 1;
    constexpr auto N_BLUE_BITMAP_PIXELS = BLUE_WIDTH_PX * BLUE_HEIGHT_PX + 1;
    constexpr auto N_WALL_BITMAP_PIXELS = WALL_WIDTH_PX * WALL_HEIGHT_PX + 1;

    constexpr u32 N_TILE_ASSETS = 2;
    constexpr u32 N_PLAYER_ASSETS = 1;
    constexpr u32 N_BLUE_ASSETS = 1;
    constexpr u32 N_WALL_ASSETS = 1;

    constexpr size_t DeviceAssets_Pixel = 
        sizeof(Pixel) * 
        (N_TILE_ASSETS * N_WORLD_TILES + 
        N_PLAYER_ASSETS * N_PLAYER_BITMAP_PIXELS + 
        N_BLUE_ASSETS * N_BLUE_BITMAP_PIXELS +
        N_WALL_ASSETS * N_WALL_BITMAP_PIXELS);
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

    Vec2Dr32 delta_pos_m;

    WorldPosition next_position;

    uStatus status = 0;
};

/*
class EntitySOA
{
public:
    u32 n_elements;
    
    r32* width_m;
    r32* height_m;
    
    Image* bitmap;
    Pixel* avg_color;

    WorldPosition* position;
    Vec2Dr32* dt;
    r32* speed;

    Vec2Dr32* delta_pos_m;

    WorldPosition* next_position;

    uStatus* status;
};


namespace SIZE
{
    constexpr size_t Entity_r32 = sizeof(r32) * 3;
    constexpr size_t Entity_Image = sizeof(Image);
    constexpr size_t Entity_Pixel = sizeof(Pixel);
    constexpr size_t Entity_WorldPosition = sizeof(WorldPosition) * 2;
    constexpr size_t Entity_Vec2Dr32 = sizeof(Vec2Dr32) * 2;
    constexpr size_t Entity_uStatus = sizeof(uStatus);
}
*/

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

    //EntitySOA entity_soa;

    Image screen_pixels;
};

namespace SIZE
{
    constexpr auto N_SCREEN_PIXELS = SCREEN_WIDTH_PX * SCREEN_WIDTH_PX;
    constexpr auto N_WORLD_TILES = WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE;

    constexpr size_t DeviceMemory_Pixel = DeviceAssets_Pixel + N_SCREEN_PIXELS * sizeof(Pixel);
    constexpr size_t DeviceMemory_Tile = N_WORLD_TILES * sizeof(Tile);
}


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
    MemoryBuffer<r32> device_r32_buffer;
    MemoryBuffer<Image> device_image_buffer;
    MemoryBuffer<Pixel> device_pixel_buffer;
    MemoryBuffer<Tile> device_tile_buffer;
    MemoryBuffer<Entity> device_entity_buffer;

    MemoryBuffer<UnifiedMemory> unified_buffer;
    MemoryBuffer<InputRecord> unified_input_record_buffer;
};