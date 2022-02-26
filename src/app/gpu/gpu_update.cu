#include "gpu_include.cuh"

#include <cassert>


constexpr auto N_PLAYER_WALL_COLLISIONS = N_PLAYERS * N_BROWN_ENTITIES;
constexpr auto N_BLUE_WALL_COLLISIONS = N_BLUE_ENTITIES * N_BROWN_ENTITIES;
constexpr auto N_PLAYER_BLUE_COLLISIONS =  N_PLAYERS * N_BLUE_ENTITIES;
constexpr auto N_BLUE_BLUE_COLLISIONS = N_BLUE_ENTITIES * N_BLUE_ENTITIES;
constexpr auto N_COLLISIONS = N_PLAYER_WALL_COLLISIONS + N_BLUE_WALL_COLLISIONS + N_PLAYER_BLUE_COLLISIONS + N_BLUE_BLUE_COLLISIONS;

namespace gpu
{
/*************************/


constexpr auto PLAYER_WALL_BEGIN = 0;
constexpr auto PLAYER_WALL_END = PLAYER_WALL_BEGIN + N_PLAYER_WALL_COLLISIONS;
constexpr auto BLUE_WALL_BEGIN = PLAYER_WALL_END;
constexpr auto BLUE_WALL_END = BLUE_WALL_BEGIN + N_BLUE_WALL_COLLISIONS;
constexpr auto PLAYER_BLUE_BEGIN = BLUE_WALL_END;
constexpr auto PLAYER_BLUE_END = PLAYER_BLUE_BEGIN + N_PLAYER_BLUE_COLLISIONS;
constexpr auto BLUE_BLUE_BEGIN = PLAYER_BLUE_END;
constexpr auto BLUE_BLUE_END = BLUE_BLUE_BEGIN + N_BLUE_BLUE_COLLISIONS;


GPU_FUNCTION
static bool is_player_wall(u32 offset)
{
    return /*offset >= PLAYER_WALL_BEGIN &&*/ offset < PLAYER_WALL_END;
}


GPU_FUNCTION
static bool is_blue_wall(u32 offset)
{
    return offset >= BLUE_WALL_BEGIN && offset < BLUE_WALL_END;
}


GPU_FUNCTION
static bool is_player_blue(u32 offset)
{
    return offset >= PLAYER_BLUE_BEGIN && offset < PLAYER_BLUE_END;
}


GPU_FUNCTION
static bool is_blue_blue(u32 offset)
{
    return offset >= BLUE_BLUE_BEGIN && offset < BLUE_BLUE_END;
}


GPU_FUNCTION
static u32 get_entity_id_from_player_offset(u32 offset)
{
    return gpu::PLAYER_BEGIN + offset;
}


GPU_FUNCTION
static u32 get_entity_id_from_blue_offset(u32 offset)
{
    return gpu::BLUE_BEGIN + offset;
}


GPU_FUNCTION
static u32 get_entity_id_from_brown_offset(u32 offset)
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


/*************************/
}


class MoveEntityProps
{
public:
    DeviceArray<Entity> entities;

    Vec2Dr32 player_dt;
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

    if(gpu::is_player_entity(entity_id))
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

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_next_positions<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


GPU_FUNCTION
static void stop_wall(Entity& ent, Entity const& wall)
{   
    if(!ent.is_active || !wall.is_active)
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
static void bounce_wall(Entity& ent, Entity const& wall)
{
    if(!ent.is_active || !wall.is_active)
    {
        return;
    }

    if(ent.delta_pos_m.x == 0.0f && ent.delta_pos_m.y == 0.0f)
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

    auto e_x_finish = gpu::add_delta(e_start, { ent.delta_pos_m.x, 0.0f });
    if(gpu::rect_intersect(e_x_finish, w))
    {
        ent.inv_x = true;
    }

    auto e_y_finish = gpu::add_delta(e_start, { 0.0f, ent.delta_pos_m.y });
    if(gpu::rect_intersect(e_y_finish, w))
    {
        ent.inv_y = true;
    }
    
}


GPU_FUNCTION
static void blue_blue(Entity& a, Entity const& b)
{
    if(!a.is_active || !b.is_active)
    {
        return;
    }

    if(a.delta_pos_m.x == 0.0f && a.delta_pos_m.y == 0.0f)
    {
        return;
    }

    auto delta = gpu::sub_delta_m(a.position, b.next_position);
    
    auto b_finish = gpu::make_rect(b.width, b.height);
    auto a_start = gpu::make_rect(delta, a.width, a.height);
    auto a_finish = gpu::add_delta(a_start, a.delta_pos_m);

    if(!gpu::rect_intersect(a_finish, b_finish))
    {
        return;
    }

    auto a_x_finish = gpu::add_delta(a_start, { a.delta_pos_m.x, 0.0f });
    if(gpu::rect_intersect(a_x_finish, b_finish))
    {
        a.inv_x = true;
    }

    auto a_y_finish = gpu::add_delta(a_start, { 0.0f, a.delta_pos_m.y });
    if(gpu::rect_intersect(a_y_finish, b_finish))
    {
        a.inv_y = true;
    }
}


GPU_FUNCTION
static void player_blue(Entity const& player, Entity& blue)
{   
    if(!player.is_active || !blue.is_active || !gpu::entity_will_intersect(player, blue))
    {
        return;
    }
    /*

    if(!gpu::equal(player.dt, { 0.0f, 0.0f }))
    {
        blue.dt = player.dt;
        blue.speed = player.speed;
        blue.delta_pos_m = player.delta_pos_m;
    }
    else
    {
        blue_blue(blue, player);
    }
    */

    //blue_blue(blue, player);
    blue.is_active = false;
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

        stop_wall(player, wall);

        return;
    }
    else if(gpu::is_blue_wall(collision_offset))
    {
        auto offset = collision_offset - gpu::BLUE_WALL_BEGIN;
        auto blue_offset = offset / N_BROWN_ENTITIES;
        auto wall_offset = offset - blue_offset * N_BROWN_ENTITIES;

        auto& blue = entities.data[gpu::get_entity_id_from_blue_offset(blue_offset)];
        auto& wall = entities.data[gpu::get_entity_id_from_brown_offset(wall_offset)];

        bounce_wall(blue, wall);

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
    else if(gpu::is_blue_blue(collision_offset))
    {
        auto offset = collision_offset - gpu::BLUE_BLUE_BEGIN;
        auto a_offset = offset / N_BLUE_ENTITIES;
        auto b_offset = offset - a_offset * N_BLUE_ENTITIES;

        if(a_offset == b_offset)
        {
            return;
        }

        auto& a = entities.data[gpu::get_entity_id_from_blue_offset(a_offset)];
        auto& b = entities.data[gpu::get_entity_id_from_blue_offset(b_offset)];

        blue_blue(a, b);

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

    if(entity.inv_x)
    {
        entity.delta_pos_m.x = 0.0f;
        entity.dt.x *= -1.0f;
    }

    if(entity.inv_y)
    {
        entity.delta_pos_m.y = 0.0f;
        entity.dt.y *= -1.0f;
    }    

    entity.position = gpu::add_delta(entity.position, entity.delta_pos_m);

    entity.next_position = entity.position;
    entity.delta_pos_m = { 0.0f, 0.0f };
    entity.inv_x = false;
    entity.inv_y = false;
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