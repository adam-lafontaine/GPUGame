#include "gpu_include.cuh"

#include <cassert>


GPU_KERNAL
void gpu_init_tiles(DeviceTileMatrix world_tiles, TileList tiles, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto world_tile_id = (u32)t;

    assert(world_tile_id < world_tiles.width * world_tiles.height);

    //auto& world_tile = world_tiles.data[world_tile_id];

    world_tiles.data[world_tile_id] = tiles.grass;

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


GPU_FUNCTION
static void init_player(Entity& player)
{
    player.is_active = true;

    player.width = 0.3f;
    player.height = 0.3f;

    player.color = gpu::to_pixel(255, 0, 0);

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

    entity.color = gpu::to_pixel(0, 0, 100);

    entity.position.tile = { 6, (i32)id + 1 };
    entity.position.offset_m = { 0.2f, 0.2f };

    entity.next_position = entity.position;

    entity.speed = 1.0f;
    entity.dt = { 0.0f, 0.0f };

    entity.delta_pos_m = { 0.0f, 0.0f };
}


GPU_FUNCTION
static void init_wall(Entity& wall, u32 wall_id)
{
    assert(wall_id < N_BROWN_ENTITIES);

    wall.is_active = true;

    wall.width = TILE_LENGTH_M;
    wall.height = TILE_LENGTH_M;

    wall.color = gpu::to_pixel(150, 75, 0);
    
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


GPU_KERNAL
void gpu_init_entities(DeviceArray<Entity> entities, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == entities.n_elements);

    auto entity_id = (u32)t;

    if(gpu::is_player_entity(entity_id))
    {        
        init_player(entities.data[entity_id]);
    }
    else if(gpu::is_blue_entity(entity_id))
    {
        init_entity(entities.data[entity_id], gpu::get_blue_offset(entity_id));
    }
    else if(gpu::is_brown_entity(entity_id))
    {        
        init_wall(entities.data[entity_id], gpu::get_brown_offset(entity_id));
    }
}


GPU_KERNAL
void gpu_empty_kernel(u32 n_threads)
{

}



namespace gpu
{
    void init_device_memory(DeviceMemory const& device)
    {
        bool proc = cuda_no_errors();
        assert(proc);

        u32 n_threads = device.tilemap.width * device.tilemap.height;
        gpu_init_tiles<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(device.tilemap, device.tile_assets, n_threads);

        proc &= cuda_launch_success();
        assert(proc);

        n_threads = device.entities.n_elements;
        gpu_init_entities<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(device.entities, n_threads);

        proc &= cuda_launch_success();
        assert(proc);
    }
}