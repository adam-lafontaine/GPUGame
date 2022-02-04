#pragma once

#include "app_types.hpp"

constexpr r32 TILE_LENGTH_M = 0.5f;

constexpr u32 MIN_SCREEN_WIDTH_TILE = 10;
constexpr u32 MAX_SCREEN_WIDTH_TILE = 100;

constexpr u32 WORLD_WIDTH_TILE = 500;
constexpr u32 WORLD_HEIGHT_TILE = 500;

constexpr WorldPosition world_minimum()
{
    WorldPosition pos{};
    pos.tile = { 0, 0 };
    pos.offset_m = { 0.0f, 0.0f };

    return pos;
}


constexpr WorldPosition world_maximum()
{
    WorldPosition pos{};
    pos.tile = { WORLD_WIDTH_TILE - 1, WORLD_HEIGHT_TILE - 1 };
    pos.offset_m = { TILE_LENGTH_M - 0.0001, TILE_LENGTH_M - 0.0001 };

    return pos;
}


inline constexpr r32 tile_distance_m(u32 n_tiles)
{
    return n_tiles * TILE_LENGTH_M;
}


constexpr r32 MIN_SCREEN_WIDTH_M = tile_distance_m(MIN_SCREEN_WIDTH_TILE);
constexpr r32 MAX_SCREEN_WIDTH_M = tile_distance_m(MAX_SCREEN_WIDTH_TILE);


constexpr u32 N_ENTITIES = 1;

constexpr u32 PLAYER_ID = 0;



void init_device_memory(DeviceMemory const& device);

void render(AppState& state);