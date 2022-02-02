#pragma once

#include "app_types.hpp"

constexpr r32 TILE_LENGTH_M = 0.5f;

constexpr u32 MIN_SCREEN_WIDTH_TILE = 10;
constexpr u32 MAX_SCREEN_WIDTH_TILE = 100;

constexpr u32 WORLD_WIDTH_TILE = 500;
constexpr u32 WORLD_HEIGHT_TILE = 500;


inline constexpr r32 tile_distance_m(u32 n_tiles)
{
    return n_tiles * TILE_LENGTH_M;
}


constexpr r32 MIN_SCREEN_WIDTH_M = tile_distance_m(MIN_SCREEN_WIDTH_TILE);
constexpr r32 MAX_SCREEN_WIDTH_M = tile_distance_m(MAX_SCREEN_WIDTH_TILE);



void init_device_memory(DeviceMemory const& device);

void render(AppState& state);