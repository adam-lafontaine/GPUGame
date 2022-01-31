#pragma once

#include "../device/device.hpp"

class Tile
{
public:
    Point2Du32 position;

    Pixel color;
};


class DeviceMemory
{
public:
    DeviceBuffer buffer;

    DeviceArray<r32> r32_array; // just because

    DeviceArray<Tile> world_tiles;
    
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
    u8 red; // just because
    u8 green;
    u8 blue;

    u32 screen_width_m;
    
};


class AppState
{
public:
    
    DeviceMemory device;
    UnifiedMemory unified;

    HostMemory host;

    StateProps props;    
};
