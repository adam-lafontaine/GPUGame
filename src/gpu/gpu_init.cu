#include "gpu_include.cuh"

#include <cassert>


namespace gpuf
{
/********************************/

GPU_FUNCTION
static void init_player(PlayerProps const& player, PlayerBitmap const& bitmap)
{
    assert(player.id < COUNT::PLAYER_ENTITIES);

    auto i = player.id;


    gpuf::set_active(player.props.status[i]);

    player.props.bitmap[i].width = bitmap.width;
    player.props.bitmap[i].height = bitmap.height;
    player.props.bitmap[i].bitmap_data = bitmap.bitmap_data;    
    player.props.bitmap[i].avg_color = *bitmap.avg_color;

    player.props.dim_m[i] = { 0.3f, 0.3f };

    player.props.position[i].tile = { 4, 4 };
    player.props.position[i].offset_m = { 0.0f, 0.0f };

    player.props.next_position[i] = player.props.position[i];

    player.props.speed[i] = 1.5f;
    player.props.dt[i] = { 0.0f, 0.0f };

    player.props.delta_pos_m[i] = { 0.0f, 0.0f };

}


GPU_FUNCTION
static void init_blue(BlueProps const& blue, BlueBitmap const& bitmap)
{
    assert(blue.id < COUNT::BLUE_ENTITIES);

    auto i = blue.id;

    gpuf::set_active(blue.props.status[i]);

    blue.props.bitmap[i].width = bitmap.width;
    blue.props.bitmap[i].height = bitmap.height;
    blue.props.bitmap[i].bitmap_data = bitmap.bitmap_data;
    blue.props.bitmap[i].avg_color = *bitmap.avg_color;

    blue.props.dim_m[i] = { 0.1f, 0.1f };

    auto w = (i32)COUNT::BLUE_W;

    auto y = (i32)i / w;
    auto x = (i32)i - y * w;

    blue.props.position[i].tile = { x + 6, y + 2 };
    blue.props.position[i].offset_m = { 0.2f, 0.2f };

    blue.props.next_position[i] = blue.props.position[i];

    blue.props.speed[i] = 3.0f;    

    blue.props.delta_pos_m[i] = { 0.0f, 0.0f };

    Vec2Dr32 dt = { 0.0f, 0.0f };
    
    switch(i % 8)
    {
        case 0:
        dt = { 1.0f, 0.0f };

        break;

        case 1:
        dt = { 0.707107f, 0.707107f };

        break;

        case 2:
        dt = { 0.0f, 1.0f };

        break;

        case 3:
        dt = { -0.707107f, 0.707107f };

        break;

        case 4:
        dt = { -1.0f, 0.0f };

        break;

        case 5:
        dt = { -0.707107f, -0.707107f };

        break;

        case 6:
        dt = { 0.0f, -1.0f };

        break;

        case 7:
        dt = { 0.707107f, -0.707107f };

        break;
    }

    blue.props.dt[i] = gpuf::vec_mul(dt, 1.0f / 60.0f); // assume 60 FPS
}


GPU_FUNCTION
static void init_wall(WallProps const& wall, WallBitmap const& bitmap)
{
    assert(wall.id < COUNT::WALL_ENTITIES);

    auto i = wall.id;
    
    gpuf::set_active(wall.props.status[i]);

    wall.props.bitmap[i].width = bitmap.width;
    wall.props.bitmap[i].height = bitmap.height;
    wall.props.bitmap[i].bitmap_data = bitmap.bitmap_data;    
    wall.props.bitmap[i].avg_color = *bitmap.avg_color;

    wall.props.dim_m[i] = { TILE_LENGTH_M, TILE_LENGTH_M };
    
    wall.props.position[i].offset_m = { 0.0f, 0.0f };

    i32 x = 0;
    i32 y = 0;

    if(i < WORLD_WIDTH_TILE)
    {
        x = (i32)i;
        y = 0;
    }
    else if(i < 2 * WORLD_WIDTH_TILE)
    {
        y = (i32)WORLD_HEIGHT_TILE - 1;
        x = i - (i32)WORLD_WIDTH_TILE;        
    }
    else if(i < 2 * WORLD_WIDTH_TILE + WORLD_HEIGHT_TILE - 2)
    {
        x = 0;
        y = i - (2 * WORLD_WIDTH_TILE) + 1;
    }
    else
    {
        x = (i32)WORLD_WIDTH_TILE - 1;
        y = i - (2 * WORLD_WIDTH_TILE + WORLD_HEIGHT_TILE - 2) + 1;
    }

    wall.props.position[i].tile = { x, y };
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
    
    PlayerProps player{};
    player.id = (u32)t;
    player.props = device.player_soa;
    gpuf::init_player(player, assets.player_bitmap);
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

    BlueProps blue{};
    blue.id = (u32)t;
    blue.props = device.blue_soa;
    gpuf::init_blue(blue, assets.blue_bitmap);
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
    
    WallProps wall{};
    wall.id = (u32)t;
    wall.props = device.wall_soa;
    gpuf::init_wall(wall, assets.wall_bitmap);
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