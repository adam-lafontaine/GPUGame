#include "gpu_include.cuh"

#include <cassert>





class UpdateEntityProps
{
public:
    DeviceArray<Entity> entities;

    Vec2Dr32 player_dt;
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
        entity.dt = props.player_dt;
    }
    else
    {
        //entity.dt = props.player_dt;
        //entity.speed = 0.25f;
    }

    auto& pos = entity.position;
    auto& speed = entity.speed;
    auto& dt = entity.dt;

    Vec2Dr32 delta_m = gpu::vec_mul(dt, speed);

    entity.next_position = gpu::add_delta(pos, delta_m);

    pos = entity.next_position;
}


static void update_entities(AppState& state)
{
    auto n_threads = state.device.entities.n_elements;

    UpdateEntityProps props{};
    props.entities = state.device.entities;
    props.player_dt = state.props.player_dt;

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