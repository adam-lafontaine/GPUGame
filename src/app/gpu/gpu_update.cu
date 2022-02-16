#include "gpu_include.cuh"

#include <cassert>





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

    gpu::update_position(pos, delta_m);
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
    
    void update(AppState& state)
    {        
        update_entities(state);
    }
}