#pragma once

#include "tiles/device_tile.hpp"
#include "input_record/input_record.hpp"


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


constexpr auto EntitySize = sizeof(Entity);
constexpr auto WorldPositionSize = sizeof(WorldPosition);
constexpr auto Vec2Dr32Size = sizeof(Vec2Dr32);



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

using DeviceTileMatrix = DeviceMatrix<DeviceTile>;


class DeviceAssets
{
public:
    DeviceTile grass_tile;
    
    DeviceTile brown_tile;
    DeviceTile black_tile;
};


class DeviceMemory
{
public:

    DeviceAssets assets;

    Entity user_player;
    
    DeviceTileMatrix tilemap;
    
    
    DeviceEntityArray blue_entities;

    DeviceEntityArray wall_entities;    

    // will fail to run if this not here
    DeviceEntityArray memory_bug;
};


class UnifiedMemory
{
public:

    DeviceImage screen_pixels;

    DeviceInputList previous_inputs;
    
    DeviceInputList current_inputs;
    
    u64 frame_count;
};


using HostImage = Image;


class HostMemory
{
public:

    //HostImage screen_pixels;

    
};


class StateProps
{
public:

    bool reset_frame_count;

    u32 screen_width_px;
    u32 screen_height_px;

    r32 screen_width_m;

    WorldPosition screen_position;
};


class AppState
{
public:

    device::MemoryBuffer device_buffer;
    device::MemoryBuffer unified_buffer;

    DeviceMemory* device;
    UnifiedMemory* unified;

    HostMemory host;
    StateProps props;
};


size_t device_memory_total_size();

size_t unified_memory_total_size(u32 screen_width, u32 screen_height);

bool make_device_memory(DeviceMemory& memory, device::MemoryBuffer& buffer);

bool make_unified_memory(UnifiedMemory& memory, device::MemoryBuffer& buffer, u32 screen_width, u32 screen_height);