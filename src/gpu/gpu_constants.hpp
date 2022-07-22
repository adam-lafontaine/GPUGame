#pragma once


constexpr r32 TILE_LENGTH_M = 0.5f;

constexpr u32 MIN_SCREEN_WIDTH_TILE = 10;
constexpr u32 MAX_SCREEN_WIDTH_TILE = 100;

constexpr u32 WORLD_WIDTH_TILE = 50;
constexpr u32 WORLD_HEIGHT_TILE = 25;
constexpr u32 N_WORLD_TILES = WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE;

constexpr r32 MIN_SCREEN_WIDTH_M = MIN_SCREEN_WIDTH_TILE * TILE_LENGTH_M;
constexpr r32 MAX_SCREEN_WIDTH_M = MAX_SCREEN_WIDTH_TILE * TILE_LENGTH_M;

constexpr u32 N_BROWN_ENTITIES = 2 * WORLD_WIDTH_TILE + 2 * WORLD_HEIGHT_TILE - 4;

constexpr u32 N_BLUE_W = WORLD_WIDTH_TILE - 8;
constexpr u32 N_BLUE_H = WORLD_HEIGHT_TILE - 4;
constexpr u32 N_BLUE_ENTITIES = N_BLUE_W * N_BLUE_H;

constexpr u32 N_PLAYER_ENTITIES = 1;

constexpr u32 N_ENTITIES = N_BLUE_ENTITIES + N_BROWN_ENTITIES + N_PLAYER_ENTITIES;

constexpr u32 PLAYER_ID = 0;


constexpr u32 player_id(u32 player_offset)
{
    return player_offset;
}


constexpr u32 brown_id(u32 brown_offset)
{
    return brown_offset + N_PLAYER_ENTITIES;
}


constexpr u32 blue_id(u32 blue_offset)
{
    return blue_offset + N_PLAYER_ENTITIES + N_BROWN_ENTITIES;
}