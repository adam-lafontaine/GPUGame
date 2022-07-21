#include "gpu_include.cuh"

#include <cassert>


namespace gpuf
{
/********************************/


GPU_FUNCTION
static void init_player(Entity& player, PlayerBitmap const& bitmap)
{
    player.is_active = true;

    player.width = 0.3f;
    player.height = 0.3f;

    player.bitmap.data = bitmap.bitmap_data;
    player.bitmap.width = bitmap.width;
    player.bitmap.height = bitmap.height;
    player.avg_color = *bitmap.avg_color;

    player.position.tile = { 4, 4 };
    player.position.offset_m = { 0.0f, 0.0f };

    player.next_position = player.position;

    player.speed = 1.5f;
    player.dt = { 0.0f, 0.0f };

    player.delta_pos_m = { 0.0f, 0.0f };
}


GPU_FUNCTION
static void init_blue(Entity& entity, BlueBitmap const& bitmap, u32 id)
{
    assert(id < N_BLUE_ENTITIES);

    entity.is_active = true;

    entity.width = 0.1f;
    entity.height = 0.1f;

    entity.bitmap.data = bitmap.bitmap_data;
    entity.bitmap.width = bitmap.width;
    entity.bitmap.height = bitmap.height;
    entity.avg_color = *bitmap.avg_color;

    auto w = (i32)N_BLUE_W;

    auto y = (i32)id / w;
    auto x = (i32)id - y * w;

    entity.position.tile = { x + 6, y + 2 };
    entity.position.offset_m = { 0.2f, 0.2f };

    entity.next_position = entity.position;

    entity.speed = 3.0f;    

    entity.delta_pos_m = { 0.0f, 0.0f };

    entity.dt = { 0.0f, 0.0f };

    switch(id % 8)
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

    entity.dt = gpuf::vec_mul(entity.dt, 1.0f / 60.0f);
}


GPU_FUNCTION
static void init_wall(Entity& wall, WallBitmap const& bitmap, u32 wall_id)
{
    assert(wall_id < N_BROWN_ENTITIES);

    wall.is_active = true;

    wall.width = TILE_LENGTH_M;
    wall.height = TILE_LENGTH_M;

    wall.bitmap.data = bitmap.bitmap_data;
    wall.bitmap.width = bitmap.width;
    wall.bitmap.height = bitmap.height;
    wall.avg_color = *bitmap.avg_color;
    
    wall.position.offset_m = { 0.0f, 0.0f };

    wall.speed = 0.0f;
    wall.dt = { 0.0f, 0.0f };

    i32 x = 0;
    i32 y = 0;

    if(wall_id < WORLD_WIDTH_TILE)
    {
        x = (i32)wall_id;
        y = 0;
    }
    else if(wall_id < 2 * WORLD_WIDTH_TILE)
    {
        y = (i32)WORLD_HEIGHT_TILE - 1;
        x = wall_id - (i32)WORLD_WIDTH_TILE;        
    }
    else if(wall_id < 2 * WORLD_WIDTH_TILE + WORLD_HEIGHT_TILE - 2)
    {
        x = 0;
        y = wall_id - (2 * WORLD_WIDTH_TILE) + 1;
    }
    else
    {
        x = (i32)WORLD_WIDTH_TILE - 1;
        y = wall_id - (2 * WORLD_WIDTH_TILE + WORLD_HEIGHT_TILE - 2) + 1;
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

    auto& device = *device_p;
    auto& assets = device.assets;

    assert(n_threads == N_PLAYER_ENTITIES);

    gpuf::init_player(device.user_player, assets.player_bitmap);
}


GPU_KERNAL
static void gpu_init_blue_entities(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *device_p;
    auto& assets = device.assets;

    assert(n_threads == N_BLUE_ENTITIES);

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

    assert(n_threads == N_BROWN_ENTITIES);

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

        constexpr auto player_threads = N_PLAYER_ENTITIES;
        constexpr auto player_blocks = calc_thread_blocks(player_threads);

        constexpr auto blue_threads = N_BLUE_ENTITIES;
        constexpr auto blue_blocks = calc_thread_blocks(blue_threads);

        constexpr auto wall_threads = N_BROWN_ENTITIES;
        constexpr auto wall_blocks = calc_thread_blocks(wall_threads);

        constexpr auto tile_threads = N_WORLD_TILES;
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