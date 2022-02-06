#pragma once

#include "../app_types.hpp"

constexpr r32 TILE_LENGTH_M = 0.5f;

constexpr u32 MIN_SCREEN_WIDTH_TILE = 10;
constexpr u32 MAX_SCREEN_WIDTH_TILE = 100;

constexpr u32 WORLD_WIDTH_TILE = 100;
constexpr u32 WORLD_HEIGHT_TILE = 100;

constexpr r32 MIN_SCREEN_WIDTH_M = MIN_SCREEN_WIDTH_TILE * TILE_LENGTH_M;
constexpr r32 MAX_SCREEN_WIDTH_M = MAX_SCREEN_WIDTH_TILE * TILE_LENGTH_M;


constexpr u32 N_ENTITIES = 1;

constexpr u32 PLAYER_ID = 0;



namespace gpu
{
    void init_device_memory(DeviceMemory const& device);

    void update(AppState& state);

    void render(AppState& state);
}
