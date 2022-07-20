#pragma once


#include "../device/device.hpp"

using uInput = u32;

namespace INPUT
{
    constexpr uInput PLAYER_UP    = 0b0000'0000'0000'0000'0000'0000'0000'0001;
    constexpr uInput PLAYER_DOWN  = 0b0000'0000'0000'0000'0000'0000'0000'0010;
    constexpr uInput PLAYER_LEFT  = 0b0000'0000'0000'0000'0000'0000'0000'0100;
    constexpr uInput PLAYER_RIGHT = 0b0000'0000'0000'0000'0000'0000'0000'1000;

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


class DeviceInputList
{
public:
    u32 capacity;
    u32 size;
    u32 read_index;

    InputRecord* data;
};


constexpr u32 TILE_WIDTH_PX = 64;
constexpr u32 TILE_HEIGHT_PX = TILE_WIDTH_PX;


class DeviceTile
{
public:

    Pixel* bitmap_data;
    Pixel* avg_color;
};

// bitmap + avg_color
constexpr auto N_TILE_PIXELS = TILE_WIDTH_PX * TILE_HEIGHT_PX + 1;


class WorldPosition
{
public:
    Point2Di32 tile;
    Point2Dr32 offset_m;
};


class Entity
{
public:
    r32 width;
    r32 height;
    Pixel color;

    WorldPosition position;
    Vec2Dr32 dt;
    r32 speed;

    Vec2Dr32 delta_pos_m;

    WorldPosition next_position;

    b32 is_active = false;

    b32 inv_x = false;
    b32 inv_y = false;
};



class DeviceEntitySOA
{
public:
    u32 n_elements;

    r32* width;
    r32* height;
    Pixel* color;

    WorldPosition* position;
    Vec2Dr32* dt;
    r32* speed;

    Vec2Dr32* delta_pos_m;

    WorldPosition* next_position;

    b32* is_active;

    b32* inv_x;
    b32* inv_y;
};




using DeviceEntityArray = DeviceArray<Entity>;

using DeviceTileMatrix = Matrix<DeviceTile>;


class DeviceAssets
{
public:
    DeviceTile grass_tile;
    
    DeviceTile brown_tile;
    DeviceTile black_tile;
};


constexpr auto N_TILE_BITMAPS = sizeof(DeviceAssets) / sizeof(DeviceTile);


class AppInput
{
public:

    bool reset_frame_count;

    u32 screen_width_px;
    u32 screen_height_px;

    r32 screen_width_m;

    WorldPosition screen_position;
};


class DeviceMemory
{
public:

    DeviceAssets assets;

    DeviceTileMatrix tilemap;

    Entity user_player;   
    
    DeviceEntityArray blue_entities;

    DeviceEntityArray wall_entities;  
};


class UnifiedMemory
{
public:

    u64 frame_count;

    Image screen_pixels;

    DeviceInputList previous_inputs;
    DeviceInputList current_inputs;    
};


class AppState
{
public:
    AppInput app_input;

    MemoryBuffer<DeviceMemory> device_buffer;
    MemoryBuffer<Pixel> device_pixel_buffer;
    MemoryBuffer<DeviceTile> device_tile_buffer;
    MemoryBuffer<Entity> device_entity_buffer;

    MemoryBuffer<UnifiedMemory> unified_buffer;
    MemoryBuffer<Pixel> unified_pixel_buffer;
    MemoryBuffer<InputRecord> unified_input_record_buffer;
};