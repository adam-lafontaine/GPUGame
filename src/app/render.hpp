#pragma once

#include "app_types.hpp"

constexpr r32 TILE_LENGTH_M = 0.5f;

constexpr u32 MIN_SCREEN_WIDTH_TILE = 20;
constexpr u32 MAX_SCREEN_WIDTH_TILE = 200;

constexpr u32 WORLD_WIDTH_TILE = 500;
constexpr u32 WORLD_HEIGHT_TILE = 500;


inline constexpr r32 tile_distance_m(u32 n_tiles)
{
    return n_tiles * TILE_LENGTH_M;
}



void render(AppState& state);