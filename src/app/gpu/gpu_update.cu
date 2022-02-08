#include "../tiles/tiles.cuh"

#include <cassert>


GPU_KERNAL
void gpu_create_tiles(DeviceArray<Pixel> bitmap_data, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(t == 0);

    create_tiles(bitmap_data);
}


GPU_KERNAL
void gpu_init_tiles(DeviceTileMatrix tiles, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto tile_id = (u32)t;

    assert(tile_id < tiles.width * tiles.height);

    auto tile_y = tile_id / tiles.width;
    auto tile_x = tile_id - tile_y * tiles.width;

    auto& tile = tiles.data[tile_id];

    if((tile_y % 2 == 0 && tile_x % 2 != 0) || (tile_y % 2 != 0 && tile_x % 2 == 0))
    {       
        tile.bitmap_data = tiles::GREEN_TILE_DATA;
    }
    else
    { 
        tile.bitmap_data = tiles::WHITE_TILE_DATA;
    }

}


GPU_FUNCTION
void init_player(Entity& player)
{
    player.width = 0.3f;
    player.height = 0.3f;

    player.color = to_pixel(255, 0, 0);

    player.position.tile = { 5, 5 };
    player.position.offset_m = { 0.2f, 0.2f };

    player.speed = 1.5f;
    player.direction = { 0.0f, 0.0f };
}


GPU_KERNAL
void gpu_init_entities(DeviceArray<Entity> entities, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto entity_id = (u32)t;

    if(entity_id == PLAYER_ID)
    {
        init_player(entities.data[entity_id]);
    }
}


class UpdateEntityProps
{
public:
    DeviceArray<Entity> entities;

    Vec2Dr32 player_direction;
};


GPU_KERNAL
static void gpu_update_entities(UpdateEntityProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto entity_id = (u32)t;
    auto& entity = props.entities.data[entity_id];

    if(entity_id == PLAYER_ID)
    {
        entity.direction = props.player_direction;
    }

    auto& pos = entity.position;
    auto& speed = entity.speed;
    auto& direction = entity.direction;

    Vec2Dr32 delta_m;
    delta_m.x = speed * direction.x;
    delta_m.y = speed * direction.y;

    update_position(pos, delta_m);
}


static void update_entities(AppState& state)
{
    auto n_threads = state.device.entities.n_elements;

    UpdateEntityProps props{};
    props.entities = state.device.entities;
    props.player_direction = state.props.player_direction;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_update_entities<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


namespace gpu
{
    void init_device_memory(DeviceMemory const& device)
    {
        bool proc = cuda_no_errors();
        assert(proc);

        u32 n_threads = 1;
        gpu_create_tiles<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(device.tile_bitmap_data, n_threads);

        proc &= cuda_launch_success();
        assert(proc);

        n_threads = device.tilemap.width * device.tilemap.height;
        gpu_init_tiles<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(device.tilemap, n_threads);

        proc &= cuda_launch_success();
        assert(proc);

        n_threads = device.entities.n_elements;
        gpu_init_entities<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(device.entities, n_threads);

        proc &= cuda_launch_success();
        assert(proc);
    }


    void update(AppState& state)
    {        
        update_entities(state);
    }
}