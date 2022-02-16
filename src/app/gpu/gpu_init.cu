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

    auto& world_tile = world_tiles.data[world_tile_id];

    auto world_tile_y = world_tile_id / world_tiles.width;
    auto world_tile_x = world_tile_id - world_tile_y * world_tiles.width;

    if( world_tile_x == 0 ||
        world_tile_x == world_tiles.width - 1 ||
        world_tile_y == 0 ||
        world_tile_y == world_tiles.height - 1)
    {
        world_tile = tiles.brown;
    }
    else
    {
        world_tile = tiles.grass;
    }
}


GPU_FUNCTION
void init_player(Entity& player)
{
    player.width = 0.3f;
    player.height = 0.3f;

    player.color = gpu::to_pixel(255, 0, 0);

    player.position.tile = { 5, 5 };
    player.position.offset_m = { 0.2f, 0.2f };

    player.speed = 1.5f;
    player.direction = { 0.0f, 0.0f };
}


GPU_FUNCTION
void init_entity(Entity& entity, i32 id)
{
    entity.width = 0.1f;
    entity.height = 0.1f;

    entity.color = gpu::to_pixel(0, 0, 100);

    entity.position.tile = { 20, id };
    entity.position.offset_m = { 0.2f, 0.2f };

    entity.speed = 1.0f;
    entity.direction = { 0.0f, 0.0f };
}


GPU_KERNAL
void gpu_init_entities(DeviceArray<Entity> entities, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto entity_id = t;

    if(entity_id == PLAYER_ID)
    {
        init_player(entities.data[entity_id]);
        return;
    }

    init_entity(entities.data[entity_id], entity_id);
}


namespace gpu
{
    void init_device_memory(DeviceMemory const& device)
    {
        bool proc = cuda_no_errors();
        assert(proc);

        u32 n_threads = device.tilemap.width * device.tilemap.height;
        gpu_init_tiles<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(device.tilemap, device.tiles, n_threads);

        proc &= cuda_launch_success();
        assert(proc);

        n_threads = device.entities.n_elements;
        gpu_init_entities<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(device.entities, n_threads);

        proc &= cuda_launch_success();
        assert(proc);
    }
}