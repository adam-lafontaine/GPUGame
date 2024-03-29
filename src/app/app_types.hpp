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

    f32 est_dt_frame;
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


class EntityBitmap
{
public:
    Pixel* bitmap_data;
    Pixel avg_color;

    u32 width;
    u32 height;
};



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
    Point2Df32 offset_m;
};


using uStatus = u32;

namespace STATUS
{
    constexpr uStatus ACTIVE = 1;
    constexpr uStatus ONSCREEN = 2 * ACTIVE;
    constexpr uStatus INV_X = 2 * ONSCREEN;
    constexpr uStatus INV_Y = 2 * INV_X;
    constexpr uStatus STOP_X = 2 * INV_Y;
    constexpr uStatus STOP_Y = 2 * STOP_X;
}


using TileMatrix = Matrix<Tile>;


class PlayerEntitySOA
{
public:

    uStatus* status;

    EntityBitmap* bitmap;

    Vec2Df32* dim_m;

    WorldPosition* position;
    Vec2Df32* dt;
    f32* speed;

    Vec2Df32* delta_pos_m;

    WorldPosition* next_position;
};


namespace SIZE
{
    constexpr size_t PlayerEntitySOA_uStatus = COUNT::PLAYER_ENTITIES * sizeof(uStatus);
    constexpr size_t PlayerEntitySOA_EntityBitmap = COUNT::PLAYER_ENTITIES * sizeof(EntityBitmap);
    constexpr size_t PlayerEntitySOA_Vec2Df32 = COUNT::PLAYER_ENTITIES * sizeof(Vec2Df32) * 3;
    constexpr size_t PlayerEntitySOA_WorldPosition = COUNT::PLAYER_ENTITIES * sizeof(WorldPosition) * 2;
    constexpr size_t PlayerEntitySOA_f32 = COUNT::PLAYER_ENTITIES * sizeof(f32);
}


class BlueEntitySOA
{
public:

    uStatus* status;

    EntityBitmap* bitmap;

    Vec2Df32* dim_m;

    WorldPosition* position;

    Vec2Df32* dt;
    f32* speed;

    Vec2Df32* delta_pos_m;

    WorldPosition* next_position;
};


namespace SIZE
{
    constexpr size_t BlueEntitySOA_uStatus = COUNT::BLUE_ENTITIES * sizeof(uStatus);
    constexpr size_t BlueEntitySOA_EntityBitmap = COUNT::BLUE_ENTITIES * sizeof(EntityBitmap);
    constexpr size_t BlueEntitySOA_Vec2Df32 = COUNT::BLUE_ENTITIES * sizeof(Vec2Df32) * 3;
    constexpr size_t BlueEntitySOA_WorldPosition = COUNT::BLUE_ENTITIES * sizeof(WorldPosition) * 2;
    constexpr size_t BlueEntitySOA_f32 = COUNT::BLUE_ENTITIES * sizeof(f32);
}


class WallEntitySOA
{
public:

    uStatus* status;

    EntityBitmap* bitmap;

    Vec2Df32* dim_m;

    WorldPosition* position;
};


namespace SIZE
{
    constexpr size_t WallEntitySOA_uStatus = COUNT::WALL_ENTITIES * sizeof(uStatus);
    constexpr size_t WallEntitySOA_Image = COUNT::WALL_ENTITIES * sizeof(Image);
    constexpr size_t WallEntitySOA_Pixel = COUNT::WALL_ENTITIES * sizeof(Pixel);
    constexpr size_t WallEntitySOA_EntityBitmap = COUNT::PLAYER_ENTITIES * sizeof(EntityBitmap);
    constexpr size_t WallEntitySOA_Vec2Df32 = COUNT::WALL_ENTITIES * sizeof(Vec2Df32);
    constexpr size_t WallEntitySOA_WorldPosition = COUNT::WALL_ENTITIES * sizeof(WorldPosition);
}


namespace SIZE
{
    constexpr size_t Entity_uStatus = PlayerEntitySOA_uStatus + BlueEntitySOA_uStatus + WallEntitySOA_uStatus;
    constexpr size_t Entity_EntityBitmap = PlayerEntitySOA_EntityBitmap + BlueEntitySOA_EntityBitmap + WallEntitySOA_EntityBitmap;
    constexpr size_t Entity_WorldPosition = PlayerEntitySOA_WorldPosition + BlueEntitySOA_WorldPosition + WallEntitySOA_WorldPosition;
    constexpr size_t Entity_Vec2Df32 = PlayerEntitySOA_Vec2Df32 + BlueEntitySOA_Vec2Df32 + WallEntitySOA_Vec2Df32;
    constexpr size_t Entity_f32 = PlayerEntitySOA_f32 + BlueEntitySOA_f32;
}


class AppInput
{
public:

    bool reset_frame_count;

    f32 screen_width_m;
    f32 screen_height_m;

    WorldPosition screen_position;
};


class DeviceMemory
{
public:

    DeviceAssets assets;

    TileMatrix tilemap;
    
    Image screen_pixels;

    PlayerEntitySOA player_soa;
    BlueEntitySOA blue_soa;
    WallEntitySOA wall_soa;
};


namespace SIZE
{
    constexpr size_t DeviceMemory_Pixel = DeviceAssets_Pixel + COUNT::SCREEN_PIXELS * sizeof(Pixel);
    constexpr size_t DeviceMemory_Tile = COUNT::WORLD_TILES * sizeof(Tile);
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

    MemoryBuffer<UnifiedMemory> unified_buffer;
    MemoryBuffer<InputRecord> unified_input_record_buffer;

    // soa data
    MemoryBuffer<uStatus> device_entity_ustatus_buffer;
    MemoryBuffer<EntityBitmap> device_entity_entity_bitmap_buffer;
    MemoryBuffer<Rect2Df32> device_entity_rect_2d_f32_buffer;
    MemoryBuffer<f32> device_entity_f32_buffer;
    MemoryBuffer<WorldPosition> device_entity_world_position_buffer;
    MemoryBuffer<Vec2Df32> device_entity_vec_2d_f32_buffer;
};