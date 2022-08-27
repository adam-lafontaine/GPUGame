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
        (N_TILE_ASSETS * COUNT::WORLD_TILES + 
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

/*
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


using EntityArray = Array<Entity>;
*/
using TileMatrix = Matrix<Tile>;


class PlayerEntitySOA
{
public:

    uStatus* status;

    Image* bitmap;
    Pixel* avg_color;

    Vec2Dr32* dim_m;

    WorldPosition* position;
    Vec2Dr32* dt;
    r32* speed;

    Vec2Dr32* delta_pos_m;

    WorldPosition* next_position;
};


namespace SIZE
{
    constexpr size_t PlayerEntitySOA_uStatus = COUNT::PLAYER_ENTITIES * sizeof(uStatus);
    constexpr size_t PlayerEntitySOA_Image = COUNT::PLAYER_ENTITIES * sizeof(Image);
    constexpr size_t PlayerEntitySOA_Pixel = COUNT::PLAYER_ENTITIES * sizeof(Pixel);
    constexpr size_t PlayerEntitySOA_Vec2Dr32 = COUNT::PLAYER_ENTITIES * sizeof(Vec2Dr32) * 3;
    constexpr size_t PlayerEntitySOA_WorldPosition = COUNT::PLAYER_ENTITIES * sizeof(WorldPosition) * 2;
    constexpr size_t PlayerEntitySOA_r32 = COUNT::PLAYER_ENTITIES * sizeof(r32);
}


class BlueEntitySOA
{
public:

    uStatus* status;

    Image* bitmap;
    Pixel* avg_color;

    Vec2Dr32* dim_m;

    WorldPosition* position;

    Vec2Dr32* dt;
    r32* speed;

    Vec2Dr32* delta_pos_m;

    WorldPosition* next_position;
};


namespace SIZE
{
    constexpr size_t BlueEntitySOA_uStatus = COUNT::BLUE_ENTITIES * sizeof(uStatus);
    constexpr size_t BlueEntitySOA_Image = COUNT::BLUE_ENTITIES * sizeof(Image);
    constexpr size_t BlueEntitySOA_Pixel = COUNT::BLUE_ENTITIES * sizeof(Pixel);
    constexpr size_t BlueEntitySOA_Vec2Dr32 = COUNT::BLUE_ENTITIES * sizeof(Vec2Dr32) * 3;
    constexpr size_t BlueEntitySOA_WorldPosition = COUNT::BLUE_ENTITIES * sizeof(WorldPosition) * 2;
    constexpr size_t BlueEntitySOA_r32 = COUNT::BLUE_ENTITIES * sizeof(r32);
}


class WallEntitySOA
{
public:

    uStatus* status;

    Image* bitmap;
    Pixel* avg_color;

    Vec2Dr32* dim_m;

    WorldPosition* position;
};


namespace SIZE
{
    constexpr size_t WallEntitySOA_uStatus = COUNT::WALL_ENTITIES * sizeof(uStatus);
    constexpr size_t WallEntitySOA_Image = COUNT::WALL_ENTITIES * sizeof(Image);
    constexpr size_t WallEntitySOA_Pixel = COUNT::WALL_ENTITIES * sizeof(Pixel);
    constexpr size_t WallEntitySOA_Vec2Dr32 = COUNT::WALL_ENTITIES * sizeof(Vec2Dr32);
    constexpr size_t WallEntitySOA_WorldPosition = COUNT::WALL_ENTITIES * sizeof(WorldPosition);
}


namespace SIZE
{
    constexpr size_t Entity_uStatus = PlayerEntitySOA_uStatus + BlueEntitySOA_uStatus + WallEntitySOA_uStatus;
    constexpr size_t Entity_Image = PlayerEntitySOA_Image + BlueEntitySOA_Image + WallEntitySOA_Image;
    constexpr size_t Entity_Pixel = PlayerEntitySOA_Pixel + BlueEntitySOA_Pixel + WallEntitySOA_Pixel;
    constexpr size_t Entity_WorldPosition = PlayerEntitySOA_WorldPosition + BlueEntitySOA_WorldPosition + WallEntitySOA_WorldPosition;
    constexpr size_t Entity_Vec2Dr32 = PlayerEntitySOA_Vec2Dr32 + BlueEntitySOA_Vec2Dr32 + WallEntitySOA_Vec2Dr32;
    constexpr size_t Entity_r32 = PlayerEntitySOA_r32 + BlueEntitySOA_r32;
}


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
/*
    EntityArray entities;

    EntityArray player_entities;    
    EntityArray blue_entities;
    EntityArray wall_entities;
*/
    Image screen_pixels;

    PlayerEntitySOA player_soa;
    BlueEntitySOA blue_soa;
    WallEntitySOA wall_soa;
};


namespace SIZE
{
    constexpr size_t DeviceMemory_Pixel = DeviceAssets_Pixel + COUNT::SCREEN_PIXELS * sizeof(Pixel);
    constexpr size_t DeviceMemory_Tile = COUNT::WORLD_TILES * sizeof(Tile);
    //constexpr size_t DeviceMemory_Entity = COUNT::ENTITIES * sizeof(Entity);
}


class UnifiedMemory
{
public:

    u64 frame_count;    

    InputList previous_inputs;
    InputList current_inputs;

    u32 user_player_id = 0;
};


class AppState
{
public:
    AppInput app_input;

    Image device_pixels;
    Image screen_pixels;

    MemoryBuffer<DeviceMemory> device_buffer;
    MemoryBuffer<Image> device_image_buffer;
    MemoryBuffer<Pixel> device_pixel_buffer;
    MemoryBuffer<Tile> device_tile_buffer;
    //MemoryBuffer<Entity> device_entity_buffer;

    MemoryBuffer<UnifiedMemory> unified_buffer;
    MemoryBuffer<InputRecord> unified_input_record_buffer;

    // soa data
    MemoryBuffer<uStatus> device_entity_ustatus_buffer;
    MemoryBuffer<Image> device_entity_image_buffer;
    MemoryBuffer<Pixel> device_entity_pixel_buffer;
    MemoryBuffer<Rect2Dr32> device_entity_rect_2d_r32_buffer;
    MemoryBuffer<r32> device_entity_r32_buffer;
    MemoryBuffer<WorldPosition> device_entity_world_position_buffer;
    MemoryBuffer<Vec2Dr32> device_entity_vec_2d_r32_buffer;
};