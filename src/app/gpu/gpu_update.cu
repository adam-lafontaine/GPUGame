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


GPU_FUNCTION
static bool entity_will_intersect(Entity const& lhs, Entity const& rhs)
{
    auto delta_m = gpu::sub_delta_m(lhs.next_position, rhs.next_position);
    
    auto rhs_rect = gpu::make_rect(rhs.width, rhs.height);
    auto lhs_rect = gpu::make_rect(delta_m, lhs.width, lhs.height);

    return gpu::rect_intersect(lhs_rect, rhs_rect);
}




/*
GPU_FUNCTION
static void collision_stop(Entity& lhs, Entity& rhs)
{
    // everything relative to rhs.position
    Point2Dr32 rhs_pt_start = { 0.0f, 0.0f };
    Point2Dr32 rhs_pt_finish = gpu::sub_delta_m(rhs.next_position, rhs.position);
    Point2Dr32 lhs_pt_start = gpu::sub_delta_m(lhs.position, rhs.position);
    Point2Dr32 lhs_pt_finish = gpu::sub_delta_m(lhs.next_position, rhs.position);

    auto delta_rhs = gpu::subtract(rhs_pt_finish, rhs_pt_start);
    auto delta_lhs = gpu::subtract(lhs_pt_finish, lhs_pt_start);

    auto rhs_rect_start = gpu::make_rect(rhs_pt_start, rhs.width, rhs.height);
    auto lhs_rect_start = gpu::make_rect(lhs_pt_start, lhs.width, lhs.height);

    assert(!gpu::rect_intersect(lhs_rect_start, rhs_rect_start));

    auto rhs_rect_finish = gpu::make_rect(rhs_pt_finish, rhs.width, rhs.height);    
    auto lhs_rect_finish = gpu::make_rect(lhs_pt_finish, lhs.width, lhs.height);    

    if(!gpu::rect_intersect(lhs_rect_finish, rhs_rect_finish))
    {
        return;
    }

    r32 dx = 1.0f;
    r32 dy = 1.0f;

    // change in x
    auto lhs_x = rhs_pt_finish.x;
    auto rhs_x = lhs_pt_finish.x;
    auto lhs_y = rhs_pt_start.y;
    auto rhs_y = lhs_pt_start.y;

    auto lhs_rect = gpu::make_rect({ lhs_x, lhs_y }, lhs.width, lhs.height);
    auto rhs_rect = gpu::make_rect({ rhs_x, rhs_y }, rhs.width, rhs.height);
    for(; gpu::rect_intersect(lhs_rect, rhs_rect) && dx >= 0.0f; dx -= 0.1f)
    {
        // largest dx (0 - 1) for no intersection

        lhs_x = lhs_pt_start.x + dx * (lhs_pt_finish.x - lhs_pt_start.x);
        rhs_x = rhs_pt_start.x + dx * (rhs_pt_finish.x - rhs_pt_start.x);
        lhs_rect = gpu::make_rect({ lhs_x, lhs_y }, lhs.width, lhs.height);
        rhs_rect = gpu::make_rect({ rhs_x, rhs_y }, rhs.width, rhs.height);
    }

    if(dx < 0.0f)
    {
        dx = 0.0f;
    }

    // change in y
    lhs_x = lhs_pt_start.x;
    rhs_x = rhs_pt_start.x;
    lhs_y = lhs_pt_finish.y;
    rhs_y = rhs_pt_finish.y;

    lhs_rect = gpu::make_rect({ lhs_x, lhs_y }, lhs.width, lhs.height);
    rhs_rect = gpu::make_rect({ rhs_x, rhs_y }, rhs.width, rhs.height);
    for(; gpu::rect_intersect(lhs_rect, rhs_rect) && dy >= 0.0f; dy -= 0.1f)
    {
        // largest dy (0 - 1) for no intersection

        lhs_y = lhs_pt_start.y + dy * (lhs_pt_finish.y - lhs_pt_start.y);
        rhs_y = rhs_pt_start.y + dy * (rhs_pt_finish.y - rhs_pt_start.y);
        lhs_rect = gpu::make_rect({ lhs_x, lhs_y }, lhs.width, lhs.height);
        rhs_rect = gpu::make_rect({ rhs_x, rhs_y }, rhs.width, rhs.height);
    }

    if(dy < 0.0f)
    {
        dy = 0.0f;
    }

    delta_lhs.x *= dx;
    delta_lhs.y *= dy;
    delta_rhs.x *= dx;
    delta_rhs.y *= dy;

    lhs.next_position = gpu::add_delta(lhs.position, delta_lhs);
    rhs.next_position = gpu::add_delta(rhs.position, delta_rhs);
}
*/


/*************************/
}


class MoveEntityProps
{
public:
    DeviceArray<Entity> entities;

    Vec2Dr32 player_dt;
    bool move_blue;
};


GPU_KERNAL
static void gpu_next_positions(MoveEntityProps props, u32 n_threads)
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

    entity.dt = { 0.0f, 0.0f };

    if(gpu::is_player_entity(entity_id) && !props.move_blue)
    {
        entity.dt = props.player_dt;
    }
    else if(gpu::is_blue_entity(entity_id) && props.move_blue)
    {
        entity.dt = props.player_dt;
    }

    entity.delta_pos_m = gpu::vec_mul(entity.dt, entity.speed);

    entity.next_position = gpu::add_delta(entity.position, entity.delta_pos_m);
}


static void next_positions(AppState& state)
{
    auto n_threads = state.device.entities.n_elements;

    MoveEntityProps props{};
    props.entities = state.device.entities;
    props.player_dt = state.props.player_dt;
    props.move_blue = state.props.move_blue;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_next_positions<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


GPU_FUNCTION
static void entity_wall(Entity& ent, Entity& wall)
{   
    if(!ent.is_active || !wall.is_active/* || !gpu::entity_will_intersect(ent, wall)*/)
    {
        return;
    }

    auto delta = gpu::sub_delta_m(ent.position, wall.position);
    
    auto w = gpu::make_rect(wall.width, wall.height);
    auto e_start = gpu::make_rect(delta, ent.width, ent.height);
    auto e_finish = gpu::add_delta(e_start, ent.delta_pos_m);

    if(!gpu::rect_intersect(e_finish, w))
    {
        return;
    }

    auto mm = 0.001f;

    auto e_x_finish = gpu::add_delta(e_start, { ent.delta_pos_m.x, 0.0f });
    if(gpu::rect_intersect(e_x_finish, w))
    {
        if(fabs(e_start.x_end - w.x_begin) < mm || fabs(e_start.x_begin - w.x_end) < mm)
        {
            ent.delta_pos_m.x = 0.0f;
        }
        else if(e_start.x_end < w.x_begin)
        {
            ent.delta_pos_m.x = w.x_begin - e_start.x_end - 0.5 * mm;
        }
        else if(e_start.x_begin > w.x_end)
        {
            ent.delta_pos_m.x = w.x_end - e_start.x_begin + 0.5 * mm;
        }
    }

    auto e_y_finish = gpu::add_delta(e_start, { 0.0f, ent.delta_pos_m.y });
    if(gpu::rect_intersect(e_y_finish, w))
    {
        if(fabs(e_start.y_end - w.y_begin) < mm || fabs(e_start.y_begin - w.y_end) < mm)
        {
            ent.delta_pos_m.y = 0.0f;
        }
        else if(e_start.y_end < w.y_begin)
        {
            ent.delta_pos_m.y = w.y_begin - e_start.y_end - 0.5 * mm;
        }
        else if(e_start.y_begin > w.y_end)
        {
            ent.delta_pos_m.y = w.y_end - e_start.y_begin + 0.5 * mm;
        }
    }
}


GPU_FUNCTION
static void player_blue(Entity const& player, Entity& blue)
{   
    if(!player.is_active || !blue.is_active || !gpu::entity_will_intersect(player, blue))
    {
        return;
    }
    
    blue.is_active = false;
    blue.delta_pos_m = { 0.0f, 0.0f };
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
    entity.position = gpu::add_delta(entity.position, entity.delta_pos_m);
    entity.next_position = entity.position;
    entity.delta_pos_m = { 0.0f, 0.0f };
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


class SpawnEntityProps
{
public:
    DeviceArray<Entity> entities;

    bool spawn_blue;
};


GPU_KERNAL
static void gpu_spawn_entities(SpawnEntityProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto entity_id = (u32)t;

    if(!gpu::is_blue_entity(entity_id))
    {
        return;
    }

    auto& entity = props.entities.data[entity_id];
    if(props.spawn_blue && !entity.is_active)
    {
        entity.is_active = true;
    }
}


static void spawn_entities(AppState& state)
{
    auto n_threads = state.device.entities.n_elements;

    SpawnEntityProps props{};
    props.entities = state.device.entities;
    props.spawn_blue = state.props.spawn_blue;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_spawn_entities<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

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

        spawn_entities(state);
    }
}