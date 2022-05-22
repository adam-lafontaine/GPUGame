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
static void init_entity(Entity& entity, u32 id)
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
void gpu_init_tiles(DeviceMemory* device_ptr, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *device_ptr;

    auto& world_tiles = device.tilemap;
    auto& assets = device.assets;

    assert(n_threads == world_tiles.width * world_tiles.height);
    assert(world_tiles.data);

    auto world_tile_id = (u32)t;

    world_tiles.data[world_tile_id] = assets.grass_tile;

    /*

    auto world_tile_y = world_tile_id / world_tiles.width;
    auto world_tile_x = world_tile_id - world_tile_y * world_tiles.width;    

    if( world_tile_x == 0 ||
        world_tile_x == world_tiles.width - 1 ||
        world_tile_y == 0 ||
        world_tile_y == world_tiles.height - 1)
    {
        world_tile = tiles.black;
    }
    else
    {
        world_tile = tiles.grass;
    }
    */
}


GPU_KERNAL
void gpu_init_entities(DeviceMemory* device_ptr, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *device_ptr;

    auto& entities = device.entities;

    assert(n_threads == entities.n_elements);

    auto entity_id = (u32)t;    

    if(gpuf::is_player_entity(entity_id))
    {        
        gpuf::init_player(entities.data[entity_id]);
    }
    else if(gpuf::is_blue_entity(entity_id))
    {
        gpuf::init_entity(entities.data[entity_id], gpuf::get_blue_offset(entity_id));
        entities.data[entity_id].color = gpuf::to_pixel(0, 0, 100);
    }
    else if(gpuf::is_brown_entity(entity_id))
    {        
        gpuf::init_wall(entities.data[entity_id], gpuf::get_brown_offset(entity_id));
    }
}


namespace gpu
{
    void init_device_memory(AppState const& state)
    {
        assert(state.device);

        bool result = cuda_no_errors("init_device_memory");
        assert(result);

        u32 n_threads = N_WORLD_TILES;
        gpu_init_tiles<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(state.device, n_threads);

        result = cuda_launch_success("gpu_init_tiles");
        assert(result);

        n_threads = N_ENTITIES;
        gpu_init_entities<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(state.device, n_threads);

        result = cuda_launch_success("gpu_init_entities");
        assert(result);
    }
}