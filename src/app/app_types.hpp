#pragma once

#include "tiles/device_tile.hpp"


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

    bool is_active = false;

    bool inv_x = false;
    bool inv_y = false;
};


using DeviceTileMatrix = DeviceMatrix<DeviceTile>;


class TileList
{
public:
    DeviceTile grass;
    
    DeviceTile brown;
    DeviceTile black;
};



class DeviceMemory
{
public:
    DeviceBuffer buffer;

    TileList tile_assets;
    
    DeviceTileMatrix tilemap;
    DeviceArray<Entity> entities;
};


class UnifiedMemory
{
public:
    DeviceBuffer buffer;

    DeviceImage screen_pixels;

    DeviceInputQueue frame_inputs;
};


class HostMemory
{
public:
    u32* elements; // just because
    u32 n_elements;
};


class StateProps
{
public:

    u64 frame_count;

    u32 screen_width_px;
    u32 screen_height_px;

    r32 screen_width_m;

    WorldPosition screen_position;
    Vec2Dr32 player_dt;
    
    bool spawn_blue;
};


class AppState
{
public:
    
    DeviceMemory device;
    UnifiedMemory unified;

    HostMemory host;

    StateProps props;    
};
