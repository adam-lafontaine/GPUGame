#pragma once

#include "../device/device.hpp"


class WorldPosition
{
public:
    Point2Di32 tile;
    Point2Dr32 offset_m;
};


class Tile
{
public:
    //Point2Du32 position;

    Pixel color;
};


class Entity
{
public:
    r32 width;
    r32 height;
    Pixel color;

    WorldPosition position;
    Vec2Dr32 direction;
    r32 speed;
};


using DeviceTileMatrix = DeviceMatrix<Tile>;


class DeviceMemory
{
public:
    DeviceBuffer buffer;
    
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
    Vec2Dr32 player_direction;
};


class AppState
{
public:
    
    DeviceMemory device;
    UnifiedMemory unified;

    HostMemory host;

    StateProps props;    
};
