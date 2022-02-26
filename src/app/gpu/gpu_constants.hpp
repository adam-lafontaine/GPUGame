#pragma once


constexpr r32 TILE_LENGTH_M = 0.5f;

constexpr u32 MIN_SCREEN_WIDTH_TILE = 10;
constexpr u32 MAX_SCREEN_WIDTH_TILE = 100;

constexpr u32 WORLD_WIDTH_TILE = 15;
constexpr u32 WORLD_HEIGHT_TILE = 15;

constexpr r32 MIN_SCREEN_WIDTH_M = MIN_SCREEN_WIDTH_TILE * TILE_LENGTH_M;
constexpr r32 MAX_SCREEN_WIDTH_M = MAX_SCREEN_WIDTH_TILE * TILE_LENGTH_M;

constexpr u32 N_BROWN_ENTITIES = 2 * WORLD_WIDTH_TILE + 2 * WORLD_HEIGHT_TILE - 4;

constexpr u32 N_BLUE_ENTITIES = 10;

constexpr u32 N_PLAYERS = 1;

constexpr u32 N_ENTITIES = N_BLUE_ENTITIES + N_BROWN_ENTITIES + N_PLAYERS;


constexpr u32 PLAYER_ID = 0;

constexpr u32 TILE_WIDTH_PX = 64;
constexpr u32 TILE_HEIGHT_PX = TILE_WIDTH_PX;
constexpr u32 N_TILE_BITMAPS = 3;
