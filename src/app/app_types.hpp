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

    WorldPosition next_position;

    bool is_active;
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

    u32 screen_width_px;
    u32 screen_height_px;

    r32 screen_width_m;

    WorldPosition screen_position;
    Vec2Dr32 player_dt;
};


class AppState
{
public:
    
    DeviceMemory device;
    UnifiedMemory unified;

    HostMemory host;

    StateProps props;    
};
