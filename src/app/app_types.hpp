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
};


class EntitySOA
{
public:
    r32* width;
    r32* height;
    Pixel* color;

    WorldPosition* position;
    Vec2Dr32* dt;
    r32* speed;

    Vec2Dr32* delta_pos_m;

    WorldPosition* next_position;

    bool* is_active;


    u32 n_elements;
};


inline size_t get_size(EntitySOA const& s)
{
    auto sz = sizeof(r32);       // width
    sz += sizeof(r32);           // height;
    sz += sizeof(Pixel);         // color
    sz += sizeof(WorldPosition); // position
    sz += sizeof(Vec2Dr32);      // dt
    sz += sizeof(r32);           // speed
    sz += sizeof(Vec2Dr32);      // delta_pos_m
    sz += sizeof(WorldPosition); // next_position
    sz += sizeof(bool);          // is_active

    return sz * s.n_elements;
}


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

    //EntitySOA entity_soa;
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
