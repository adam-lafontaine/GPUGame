#include "gpu_include.cuh"

#include <cassert>


constexpr auto N_PLAYER_WALL_COLLISIONS = N_PLAYERS * N_BROWN_ENTITIES;
constexpr auto N_BLUE_WALL_COLLISIONS = N_BLUE_ENTITIES * N_BROWN_ENTITIES;
constexpr auto N_PLAYER_BLUE_COLLISIONS = N_BLUE_ENTITIES;
constexpr auto N_COLLISIONS = N_PLAYER_WALL_COLLISIONS + N_BLUE_WALL_COLLISIONS + N_PLAYER_BLUE_COLLISIONS;

namespace gpu
{
/*************************/


GPU_CONSTEXPR_FUNCTION
static bool is_player_wall(u32 id)
{
    return id < N_PLAYER_WALL_COLLISIONS;
}


GPU_CONSTEXPR_FUNCTION
static bool is_blue_wall(u32 id)
{
    auto begin = N_PLAYER_WALL_COLLISIONS;
    auto end = begin + N_BLUE_WALL_COLLISIONS;

    return id >= begin && id < end;
}


GPU_CONSTEXPR_FUNCTION
static bool is_player_blue(u32 id)
{
    auto begin = N_PLAYER_WALL_COLLISIONS + N_BLUE_WALL_COLLISIONS;
    auto end = begin + N_PLAYER_BLUE_COLLISIONS;

    return id >= begin && id < end;
}


GPU_FUNCTION
static bool entity_intersect(Entity const& lhs, Entity const& rhs)
{
    auto delta_m = gpu::sub_delta_m(lhs.next_position, rhs.next_position);
    
    auto rhs_rect = gpu::make_rect(rhs.width, rhs.height);
    auto lhs_rect = gpu::make_rect(delta_m, lhs.width, lhs.height);

    return gpu::rect_intersect(lhs_rect, rhs_rect);
}


GPU_CONSTEXPR_FUNCTION
inline u32 get_entity_id_from_brown_id(u32 id)
{
    return N_PLAYERS + N_BLUE_ENTITIES + id;
}


/*************************/
}


class UpdateEntityProps
{
public:
    DeviceArray<Entity> entities;

    Vec2Dr32 player_dt;
};


GPU_KERNAL
static void gpu_next_positions(UpdateEntityProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto entity_id = (u32)t;

    if(gpu::is_brown_entity(entity_id))
    {
        return;
    }

    auto& entity = props.entities.data[entity_id];

    if(gpu::is_player_entity(entity_id))
    {
        entity.dt = props.player_dt;
    }

    auto& pos = entity.position;
    auto& speed = entity.speed;
    auto& dt = entity.dt;

    Vec2Dr32 delta_m = gpu::vec_mul(dt, speed);

    entity.next_position = gpu::add_delta(pos, delta_m);
}


static void next_positions(AppState& state)
{
    auto n_threads = state.device.entities.n_elements;

    UpdateEntityProps props{};
    props.entities = state.device.entities;
    props.player_dt = state.props.player_dt;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_next_positions<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


GPU_FUNCTION
static void entity_wall(Entity& ent, Entity const& wall)
{   

    if(gpu::entity_intersect(ent, wall))
    {
        ent.next_position = ent.position;
    }
    
}


GPU_KERNAL
static void gpu_collisions(DeviceArray<Entity> entities, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto collision_id = (u32)t;

    if(gpu::is_player_wall(collision_id))
    {
        auto player_id = 0; // collision_id / N_BROWN_ENTITIES;
        auto wall_id = collision_id - player_id * N_BROWN_ENTITIES;

        auto& player = entities.data[player_id];        
        auto& wall = entities.data[gpu::get_entity_id_from_brown_id(wall_id)];

        entity_wall(player, wall);

        return;
    }
    else if(gpu::is_blue_wall(collision_id))
    {

        return;
    }
    else if(gpu::is_player_blue(collision_id))
    {

        return;
    }
}


static void collisions(AppState& state)
{
    auto n_threads = N_COLLISIONS;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_collisions<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(state.device.entities, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


GPU_KERNAL
static void gpu_update_positions(DeviceArray<Entity> entities, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& entity = entities.data[t];
    entity.position = entity.next_position;
}


static void update_positions(AppState& state)
{
    auto n_threads = state.device.entities.n_elements;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_update_positions<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(state.device.entities, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


namespace gpu
{    
    void update(AppState& state)
    {        
        next_positions(state);

        collisions(state);

        update_positions(state);
    }
}