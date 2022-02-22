#include "gpu_include.cuh"

#include <cassert>


constexpr auto N_PLAYER_WALL_COLLISIONS = N_PLAYERS * N_BROWN_ENTITIES;
constexpr auto N_BLUE_WALL_COLLISIONS = N_BLUE_ENTITIES * N_BROWN_ENTITIES;
constexpr auto N_PLAYER_BLUE_COLLISIONS = N_BLUE_ENTITIES;
constexpr auto N_COLLISIONS = N_PLAYER_WALL_COLLISIONS + N_BLUE_WALL_COLLISIONS + N_PLAYER_BLUE_COLLISIONS;

namespace gpu
{
/*************************/


constexpr auto PLAYER_WALL_BEGIN = 0;
constexpr auto PLAYER_WALL_END = PLAYER_WALL_BEGIN + N_PLAYER_WALL_COLLISIONS;
constexpr auto BLUE_WALL_BEGIN = PLAYER_WALL_END;
constexpr auto BLUE_WALL_END = BLUE_WALL_BEGIN + N_BLUE_WALL_COLLISIONS;
constexpr auto PLAYER_BLUE_BEGIN = BLUE_WALL_END;
constexpr auto PLAYER_BLUE_END = PLAYER_BLUE_BEGIN + N_PLAYER_BLUE_COLLISIONS;


GPU_CONSTEXPR_FUNCTION
static bool is_player_wall(u32 offset)
{
    return /*offset >= PLAYER_WALL_BEGIN &&*/ offset < PLAYER_WALL_END;
}


GPU_CONSTEXPR_FUNCTION
static bool is_blue_wall(u32 offset)
{
    return offset >= BLUE_WALL_BEGIN && offset < BLUE_WALL_END;
}


GPU_CONSTEXPR_FUNCTION
static bool is_player_blue(u32 offset)
{
    return offset >= PLAYER_BLUE_BEGIN && offset < PLAYER_BLUE_END;
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
inline u32 get_entity_id_from_player_offset(u32 offset)
{
    return gpu::PLAYER_BEGIN + offset;
}


GPU_CONSTEXPR_FUNCTION
inline u32 get_entity_id_from_blue_offset(u32 offset)
{
    return gpu::BLUE_BEGIN + offset;
}


GPU_CONSTEXPR_FUNCTION
inline u32 get_entity_id_from_brown_offset(u32 offset)
{
    return gpu::BROWN_BEGIN + offset;
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
    else if(gpu::is_blue_entity(entity_id))
    {
        //entity.dt = props.player_dt;
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
    if(!ent.is_active || !wall.is_active)
    {
        return;
    }

    if(gpu::entity_intersect(ent, wall))
    {
        ent.next_position = ent.position;
    }    
}


GPU_FUNCTION
static void player_blue(Entity const& player, Entity& blue)
{   
    if(!player.is_active || !blue.is_active)
    {
        return;
    }

    if(gpu::entity_intersect(player, blue))
    {
        blue.is_active = false;
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

    auto collision_offset = (u32)t;

    if(gpu::is_player_wall(collision_offset))
    {
        auto offset = collision_offset - gpu::PLAYER_WALL_BEGIN;
        auto player_offset = offset / N_BROWN_ENTITIES;
        auto wall_offset = offset - player_offset * N_BROWN_ENTITIES;

        auto& player = entities.data[gpu::get_entity_id_from_player_offset(player_offset)];
        auto& wall = entities.data[gpu::get_entity_id_from_brown_offset(wall_offset)];

        entity_wall(player, wall);

        return;
    }
    else if(gpu::is_blue_wall(collision_offset))
    {
        auto offset = collision_offset - gpu::BLUE_WALL_BEGIN;
        auto blue_offset = offset / N_BROWN_ENTITIES;
        auto wall_offset = offset - blue_offset * N_BROWN_ENTITIES;

        auto& blue = entities.data[gpu::get_entity_id_from_blue_offset(blue_offset)];
        auto& wall = entities.data[gpu::get_entity_id_from_brown_offset(wall_offset)];

        entity_wall(blue, wall);

        return;
    }
    else if(gpu::is_player_blue(collision_offset))
    {
        auto offset = collision_offset - gpu::PLAYER_BLUE_BEGIN;
        auto player_offset = offset / N_BLUE_ENTITIES;
        auto blue_offset = offset - player_offset * N_BLUE_ENTITIES;

        auto& player = entities.data[gpu::get_entity_id_from_player_offset(player_offset)];
        auto& blue = entities.data[gpu::get_entity_id_from_blue_offset(blue_offset)];

        player_blue(player, blue);

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