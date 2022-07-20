#include "gpu_include.cuh"

#include <cassert>


namespace gpuf
{
/********************************/


GPU_FUNCTION
static void init_player(Entity& player)
{
    player.is_active = true;

    player.width = 0.3f;
    player.height = 0.3f;

    player.color = gpuf::to_pixel(200, 0, 0);

    player.position.tile = { 4, 4 };
    player.position.offset_m = { 0.0f, 0.0f };

    player.next_position = player.position;

    player.speed = 1.5f;
    player.dt = { 0.0f, 0.0f };

    player.delta_pos_m = { 0.0f, 0.0f };
}


GPU_FUNCTION
static void init_blue(Entity& entity, u32 id)
{
    assert(id < N_BLUE_ENTITIES);

    entity.is_active = true;

    entity.width = 0.1f;
    entity.height = 0.1f;

    entity.color = gpuf::to_pixel(0, 0, 100);

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
static void init_wall(Entity& wall, u32 wall_id)
{
    assert(wall_id < N_BROWN_ENTITIES);

    wall.is_active = true;

    wall.width = TILE_LENGTH_M;
    wall.height = TILE_LENGTH_M;

    wall.color = gpuf::to_pixel(150, 75, 0);
    
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

    auto& world_tiles = device_p->tilemap;   //.tilemap_old;
    auto& assets = device_p->assets;

    assert(n_threads == world_tiles.width * world_tiles.height);
    assert(world_tiles.data);

    auto world_tile_id = (u32)t;

    world_tiles.data[world_tile_id] = assets.grass_tile;
}


GPU_KERNAL
static void gpu_init_players(DeviceMemory* device_ptr, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *device_ptr;

    assert(n_threads == N_PLAYER_ENTITIES);

    gpuf::init_player(device.user_player);
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

    assert(n_threads == N_BLUE_ENTITIES);

    auto offset = (u32)t;

    gpuf::init_blue(device.blue_entities.data[offset], offset);
}


GPU_KERNAL
static void gpu_init_wall_entities(DeviceMemoryOld* device_ptr, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *device_ptr;

    assert(n_threads == N_BROWN_ENTITIES);

    auto offset = (u32)t;

    gpuf::init_wall(device.wall_entities_old.data[offset], offset);
}


namespace gpu
{
    bool init_device_memory(AppState const& state)
    {
        assert(state.device_old_p);

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

        auto device_p = state.device_buffer.data;
        
        
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
        
        cuda_launch_kernel(gpu_init_wall_entities, wall_blocks, THREADS_PER_BLOCK, state.device_old_p, wall_threads);
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