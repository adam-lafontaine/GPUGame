#pragma once

#include "../device/device.hpp"


class WorldPosition
{
public:
    Point2Du32 tile;
    Point2Dr32 offset_m;
};

/*
class Tile
{
public:
    Point2Du32 position;

    Pixel color;
};
*/

class DeviceMemory
{
public:
    DeviceBuffer buffer;
    
    DeviceMatrix tilemap;
    
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
    r32 screen_width_m;

    WorldPosition screen_positon;
};


class AppState
{
public:
    
    DeviceMemory device;
    UnifiedMemory unified;

    HostMemory host;

    StateProps props;    
};
