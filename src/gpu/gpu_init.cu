#include "gpu_include.cuh"

#include <cassert>


namespace gpuf
{
/********************************/



GPU_FUNCTION
static void init_player(Entity& player, PlayerBitmap const& bitmap, u32 player_offset)
{
    assert(player_offset < COUNT::PLAYER_ENTITIES);

    player.id = player_id(player_offset);
    gpuf::set_active(player);

    player.bitmap.width = bitmap.width;
    player.bitmap.height = bitmap.height;
    player.bitmap.data = bitmap.bitmap_data;
    player.avg_color = *bitmap.avg_color;

    player.width_m = 0.3f;
    player.height_m = 0.3f;

    player.position.tile = { 4, 4 };
    player.position.offset_m = { 0.0f, 0.0f };

    player.next_position = player.position;

    player.speed = 1.5f;
    player.dt = { 0.0f, 0.0f };

    player.delta_pos_m = { 0.0f, 0.0f };
}


GPU_FUNCTION
static void init_blue(Entity& entity, BlueBitmap const& bitmap, u32 blue_offset)
{
    assert(blue_offset < COUNT::BLUE_ENTITIES);

    entity.id = blue_id(blue_offset);
    gpuf::set_active(entity);

    entity.bitmap.width = bitmap.width;
    entity.bitmap.height = bitmap.height;
    entity.bitmap.data = bitmap.bitmap_data;
    entity.avg_color = *bitmap.avg_color;

    entity.width_m = 0.1f;
    entity.height_m = 0.1f;

    auto w = (i32)COUNT::BLUE_W;

    auto y = (i32)blue_offset / w;
    auto x = (i32)blue_offset - y * w;

    entity.position.tile = { x + 6, y + 2 };
    entity.position.offset_m = { 0.2f, 0.2f };

    entity.next_position = entity.position;

    entity.speed = 3.0f;    

    entity.delta_pos_m = { 0.0f, 0.0f };

    entity.dt = { 0.0f, 0.0f };

    switch(blue_offset % 8)
    {
        case 0:
        entity.dt = { 1.0f, 0.0f };

        break;

        case 1:
        entity.dt = { 0.707107f, 0.707107f };

        break;

        case 2:
        entity.dt = { 0.0f, 1.0f };

        break;

        case 3:
        entity.dt = { -0.707107f, 0.707107f };

        break;

        case 4:
        entity.dt = { -1.0f, 0.0f };

        break;

        case 5:
        entity.dt = { -0.707107f, -0.707107f };

        break;

        case 6:
        entity.dt = { 0.0f, -1.0f };

        break;

        case 7:
        entity.dt = { 0.707107f, -0.707107f };

        break;
    }

    entity.dt = gpuf::vec_mul(entity.dt, 1.0f / 60.0f); // assume 60 FPS
}


GPU_FUNCTION
static void init_wall(Entity& wall, WallBitmap const& bitmap, u32 wall_offset)
{
    assert(wall_offset < COUNT::WALL_ENTITIES);

    wall.id = wall_id(wall_offset);
    gpuf::set_active(wall);

    wall.bitmap.width = bitmap.width;
    wall.bitmap.height = bitmap.height;
    wall.bitmap.data = bitmap.bitmap_data;
    wall.avg_color = *bitmap.avg_color;

    wall.width_m = TILE_LENGTH_M;
    wall.height_m = TILE_LENGTH_M;
    
    wall.position.offset_m = { 0.0f, 0.0f };

    wall.speed = 0.0f;
    wall.dt = { 0.0f, 0.0f };

    i32 x = 0;
    i32 y = 0;

    if(wall_offset < WORLD_WIDTH_TILE)
    {
        x = (i32)wall_offset;
        y = 0;
    }
    else if(wall_offset < 2 * WORLD_WIDTH_TILE)
    {
        y = (i32)WORLD_HEIGHT_TILE - 1;
        x = wall_offset - (i32)WORLD_WIDTH_TILE;        
    }
    else if(wall_offset < 2 * WORLD_WIDTH_TILE + WORLD_HEIGHT_TILE - 2)
    {
        x = 0;
        y = wall_offset - (2 * WORLD_WIDTH_TILE) + 1;
    }
    else
    {
        x = (i32)WORLD_WIDTH_TILE - 1;
        y = wall_offset - (2 * WORLD_WIDTH_TILE + WORLD_HEIGHT_TILE - 2) + 1;
    }

    wall.position.tile = { x, y };

    wall.next_position = wall.position;

    wall.delta_pos_m = { 0.0f, 0.0f };
}


/***********************/
}


GPU_KERNAL
static void gpu_init_tiles(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& world_tiles = device_p->tilemap;
    auto& assets = device_p->assets;

    assert(n_threads == world_tiles.width * world_tiles.height);
    assert(world_tiles.data);

    auto world_tile_id = (u32)t;

    world_tiles.data[world_tile_id] = assets.grass_tile;
}


GPU_KERNAL
static void gpu_init_players(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::PLAYER_ENTITIES);

    auto& device = *device_p;
    auto& assets = device.assets;    

    auto offset = (u32)t;

    gpuf::init_player(device.player_entities.data[offset], assets.player_bitmap, (u32)t);
}


GPU_KERNAL
static void gpu_init_blue_entities(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::BLUE_ENTITIES);

    auto& device = *device_p;
    auto& assets = device.assets;

    auto offset = (u32)t;

    gpuf::init_blue(device.blue_entities.data[offset], assets.blue_bitmap, offset);
}


GPU_KERNAL
static void gpu_init_wall_entities(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *device_p;
    auto& assets = device.assets;

    assert(n_threads == COUNT::WALL_ENTITIES);

    auto offset = (u32)t;

    gpuf::init_wall(device.wall_entities.data[offset], assets.wall_bitmap, offset);
}


namespace gpu
{
    bool init_device_memory(AppState const& state)
    {
        auto device_p = state.device_buffer.data;

        assert(device_p);

        bool result = cuda::no_errors("gpu::init_device_memory");
        assert(result);
        if(!result)
        {
            return false;
        }

        constexpr auto player_threads = COUNT::PLAYER_ENTITIES;
        constexpr auto player_blocks = calc_thread_blocks(player_threads);

        constexpr auto blue_threads = COUNT::BLUE_ENTITIES;
        constexpr auto blue_blocks = calc_thread_blocks(blue_threads);

        constexpr auto wall_threads = COUNT::WALL_ENTITIES;
        constexpr auto wall_blocks = calc_thread_blocks(wall_threads);

        constexpr auto tile_threads = COUNT::WORLD_TILES;
        constexpr auto tile_blocks = calc_thread_blocks(tile_threads);        
        
        
        cuda_launch_kernel(gpu_init_players, player_blocks, THREADS_PER_BLOCK, device_p, player_threads);
        result = cuda::launch_success("gpu_init_players");
        assert(result);
        if(!result)
        {
            return false;
        }
        
        cuda_launch_kernel(gpu_init_blue_entities, blue_blocks, THREADS_PER_BLOCK, device_p, blue_threads);
        result = cuda::launch_success("gpu_init_blue_entities");
        assert(result);
        if(!result)
        {
            return false;
        }
        
        cuda_launch_kernel(gpu_init_wall_entities, wall_blocks, THREADS_PER_BLOCK, device_p, wall_threads);
        result = cuda::launch_success("gpu_init_wall_entities");
        assert(result);
        if(!result)
        {
            return false;
        }
        
        cuda_launch_kernel(gpu_init_tiles, tile_blocks, THREADS_PER_BLOCK, device_p, tile_threads);
        result = cuda::launch_success("gpu_init_tiles");
        assert(result);
        if(!result)
        {
            return false;
        }

        return true;
    }
}