#include "gpu_include.cuh"

#include <cassert>


constexpr auto N_PLAYER_WALL_COLLISIONS = N_PLAYERS * N_BROWN_ENTITIES;
constexpr auto N_BLUE_WALL_COLLISIONS = N_BLUE_ENTITIES * N_BROWN_ENTITIES;
constexpr auto N_PLAYER_BLUE_COLLISIONS =  N_PLAYERS * N_BLUE_ENTITIES;
constexpr auto N_BLUE_BLUE_COLLISIONS = N_BLUE_ENTITIES * N_BLUE_ENTITIES;
constexpr auto N_COLLISIONS = N_PLAYER_WALL_COLLISIONS + N_BLUE_WALL_COLLISIONS + N_PLAYER_BLUE_COLLISIONS + N_BLUE_BLUE_COLLISIONS;

namespace gpuf
{
/*************************/


// collision offsets
constexpr auto PLAYER_WALL_BEGIN = 0U;
constexpr auto PLAYER_WALL_END = PLAYER_WALL_BEGIN + N_PLAYER_WALL_COLLISIONS;
constexpr auto BLUE_WALL_BEGIN = PLAYER_WALL_END;
constexpr auto BLUE_WALL_END = BLUE_WALL_BEGIN + N_BLUE_WALL_COLLISIONS;
constexpr auto PLAYER_BLUE_BEGIN = BLUE_WALL_END;
constexpr auto PLAYER_BLUE_END = PLAYER_BLUE_BEGIN + N_PLAYER_BLUE_COLLISIONS;
constexpr auto BLUE_BLUE_BEGIN = PLAYER_BLUE_END;
constexpr auto BLUE_BLUE_END = BLUE_BLUE_BEGIN + N_BLUE_BLUE_COLLISIONS;


GPU_FUNCTION
static bool is_player_wall(u32 collision_offset)
{
    return /*offset >= PLAYER_WALL_BEGIN &&*/ collision_offset < PLAYER_WALL_END;
}


GPU_FUNCTION
static bool is_blue_wall(u32 collision_offset)
{
    return collision_offset >= BLUE_WALL_BEGIN && collision_offset < BLUE_WALL_END;
}


GPU_FUNCTION
static bool is_player_blue(u32 collision_offset)
{
    return collision_offset >= PLAYER_BLUE_BEGIN && collision_offset < PLAYER_BLUE_END;
}


GPU_FUNCTION
static bool is_blue_blue(u32 collision_offset)
{
    return collision_offset >= BLUE_BLUE_BEGIN && collision_offset < BLUE_BLUE_END;
}


GPU_FUNCTION
static u32 get_entity_id_from_player_offset(u32 offset)
{
    return gpuf::PLAYER_BEGIN + offset;
}


GPU_FUNCTION
static u32 get_entity_id_from_blue_offset(u32 offset)
{
    return gpuf::BLUE_BEGIN + offset;
}


GPU_FUNCTION
static u32 get_entity_id_from_brown_offset(u32 offset)
{
    return gpuf::BROWN_BEGIN + offset;
}


GPU_FUNCTION
static bool entity_will_intersect(Entity const& lhs, Entity const& rhs)
{
    auto delta_m = gpuf::sub_delta_m(lhs.next_position, rhs.next_position);
    
    auto rhs_rect = gpuf::make_rect(rhs.width, rhs.height);
    auto lhs_rect = gpuf::make_rect(delta_m, lhs.width, lhs.height);

    return gpuf::rect_intersect(lhs_rect, rhs_rect);
}


GPU_FUNCTION
static void move_player(Entity& entity, InputRecord const& input)
{    
    entity.dt = { 0.0f, 0.0f };

    if(input.input & INPUT_PLAYER_UP)
    {
        entity.dt.y -= input.est_dt_frame;
    }

    if(input.input & INPUT_PLAYER_DOWN)
    {
        entity.dt.y += input.est_dt_frame;
    }

    if(input.input & INPUT_PLAYER_LEFT)
    {
        entity.dt.x -= input.est_dt_frame;
    }

    if(input.input & INPUT_PLAYER_RIGHT)
    {
        entity.dt.x += input.est_dt_frame;
    }

    if(entity.dt.x != 0.0f && entity.dt.y != 0.0f)
    {
        entity.dt.x *= 0.707107f;
        entity.dt.y *= 0.707107f;
    }
}


GPU_FUNCTION 
void apply_current_input(Entity& entity, DeviceInputList const& inputs, u64 frame)
{
    entity.dt = { 0.0f, 0.0f };

    if(inputs.size == 0)
    {
        return;
    }

    auto& last = inputs.data[inputs.size - 1];

    auto is_last = last.frame_begin <= frame && frame < last.frame_end;

    if(!is_last)
    {
        return;
    }

    move_player(entity, last);
}


GPU_FUNCTION
void update_device_inputs(DeviceInputList const& src, DeviceInputList& dst)
{
    assert(src.data);
    assert(dst.data);

    if(src.size == 0 && dst.size == 0)
    {
        return;
    }
    if(src.size == dst.size + 1)
    {
        dst.data[dst.size++] = src.data[src.size - 1];
    }
    else if(src.size == dst.size)
    {
        dst.data[dst.size - 1] = src.data[src.size - 1];
    }
    else
    {
        assert(false);
    }    
}


GPU_FUNCTION
static void stop_wall(Entity& ent, Entity const& wall)
{   
    if(!ent.is_active || !wall.is_active)
    {
        return;
    }

    auto delta = gpuf::sub_delta_m(ent.position, wall.position);
    
    auto w = gpuf::make_rect(wall.width, wall.height);
    auto e_start = gpuf::make_rect(delta, ent.width, ent.height);
    auto e_finish = gpuf::add_delta(e_start, ent.delta_pos_m);

    if(!gpuf::rect_intersect(e_finish, w))
    {
        return;
    }

    auto mm = 0.001f;

    auto e_x_finish = gpuf::add_delta(e_start, { ent.delta_pos_m.x, 0.0f });
    if(gpuf::rect_intersect(e_x_finish, w))
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

    auto e_y_finish = gpuf::add_delta(e_start, { 0.0f, ent.delta_pos_m.y });
    if(gpuf::rect_intersect(e_y_finish, w))
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

    auto delta = gpuf::sub_delta_m(ent.position, wall.position);
    
    auto w = gpuf::make_rect(wall.width, wall.height);
    auto e_start = gpuf::make_rect(delta, ent.width, ent.height);
    auto e_finish = gpuf::add_delta(e_start, ent.delta_pos_m);

    if(!gpuf::rect_intersect(e_finish, w))
    {
        return;
    }

    auto e_x_finish = gpuf::add_delta(e_start, { ent.delta_pos_m.x, 0.0f });
    if(gpuf::rect_intersect(e_x_finish, w))
    {
        ent.inv_x = true;
    }

    auto e_y_finish = gpuf::add_delta(e_start, { 0.0f, ent.delta_pos_m.y });
    if(gpuf::rect_intersect(e_y_finish, w))
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

    auto delta = gpuf::sub_delta_m(a.position, b.next_position);
    
    auto b_finish = gpuf::make_rect(b.width, b.height);
    auto a_start = gpuf::make_rect(delta, a.width, a.height);
    auto a_finish = gpuf::add_delta(a_start, a.delta_pos_m);

    if(!gpuf::rect_intersect(a_finish, b_finish))
    {
        return;
    }

    auto a_x_finish = gpuf::add_delta(a_start, { a.delta_pos_m.x, 0.0f });
    if(gpuf::rect_intersect(a_x_finish, b_finish))
    {
        a.inv_x = true;
    }

    auto a_y_finish = gpuf::add_delta(a_start, { 0.0f, a.delta_pos_m.y });
    if(gpuf::rect_intersect(a_y_finish, b_finish))
    {
        a.inv_y = true;
    }
}


GPU_FUNCTION
static void player_blue(Entity const& player, Entity& blue)
{   
    if(!player.is_active || !blue.is_active || !gpuf::entity_will_intersect(player, blue))
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




/*************************/
}


class MoveEntityProps
{
public:
    DeviceArray<Entity> entities;
    DeviceInputList* player_inputs;
    DeviceInputList* recorded_inputs;

    u64 current_frame;
};


GPU_KERNAL
static void gpu_next_positions(MoveEntityProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == props.entities.n_elements);

    auto& player_inputs = *props.player_inputs;    

    auto entity_id = (u32)t;    

    if(gpuf::is_brown_entity(entity_id))
    {
        if(entity_id == gpuf::BROWN_BEGIN)
        {
            auto& recorded_inputs = *props.recorded_inputs;
            gpuf::update_device_inputs(player_inputs, recorded_inputs);
        }

        return;
    }

    auto& entity = props.entities.data[entity_id];

    if(gpuf::is_player_entity(entity_id))
    {
        gpuf::apply_current_input(entity, player_inputs, props.current_frame);
    }

    entity.delta_pos_m = gpuf::vec_mul(entity.dt, entity.speed);

    entity.next_position = gpuf::add_delta(entity.position, entity.delta_pos_m);
}


static void next_positions(AppState& state)
{
    auto n_threads = state.device.entities.n_elements;

    MoveEntityProps props{};
    props.entities = state.device.entities;
    props.player_inputs = state.unified.current_inputs;
    props.recorded_inputs = state.device.previous_inputs;
    props.current_frame = state.props.frame_count;

    bool proc = cuda_no_errors("next_positions");
    assert(proc);

    gpu_next_positions<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success("gpu_next_positions");
    assert(proc);
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

    if(gpuf::is_player_wall(collision_offset))
    {
        auto offset = collision_offset - gpuf::PLAYER_WALL_BEGIN;
        auto player_offset = offset / N_BROWN_ENTITIES;
        auto wall_offset = offset - player_offset * N_BROWN_ENTITIES;

        auto& player = entities.data[gpuf::get_entity_id_from_player_offset(player_offset)];
        auto& wall = entities.data[gpuf::get_entity_id_from_brown_offset(wall_offset)];

        gpuf::stop_wall(player, wall);

        return;
    }
    else if(gpuf::is_blue_wall(collision_offset))
    {
        auto offset = collision_offset - gpuf::BLUE_WALL_BEGIN;
        auto blue_offset = offset / N_BROWN_ENTITIES;
        auto wall_offset = offset - blue_offset * N_BROWN_ENTITIES;

        auto& blue = entities.data[gpuf::get_entity_id_from_blue_offset(blue_offset)];
        auto& wall = entities.data[gpuf::get_entity_id_from_brown_offset(wall_offset)];

        gpuf::bounce_wall(blue, wall);

        return;
    }
    else if(gpuf::is_player_blue(collision_offset))
    {
        auto offset = collision_offset - gpuf::PLAYER_BLUE_BEGIN;
        auto player_offset = offset / N_BLUE_ENTITIES;
        auto blue_offset = offset - player_offset * N_BLUE_ENTITIES;

        auto& player = entities.data[gpuf::get_entity_id_from_player_offset(player_offset)];
        auto& blue = entities.data[gpuf::get_entity_id_from_blue_offset(blue_offset)];

        gpuf::player_blue(player, blue);

        return;
    }
    else if(gpuf::is_blue_blue(collision_offset))
    {
        auto offset = collision_offset - gpuf::BLUE_BLUE_BEGIN;
        auto a_offset = offset / N_BLUE_ENTITIES;
        auto b_offset = offset - a_offset * N_BLUE_ENTITIES;

        if(a_offset == b_offset)
        {
            return;
        }

        auto& a = entities.data[gpuf::get_entity_id_from_blue_offset(a_offset)];
        auto& b = entities.data[gpuf::get_entity_id_from_blue_offset(b_offset)];

        gpuf::blue_blue(a, b);

        return;
    }
}


static void collisions(AppState& state)
{
    auto n_threads = N_COLLISIONS;

    bool proc = cuda_no_errors("collisions");
    assert(proc);

    gpu_collisions<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(state.device.entities, n_threads);

    proc &= cuda_launch_success("gpu_collisions");
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

    entity.position = gpuf::add_delta(entity.position, entity.delta_pos_m);

    entity.next_position = entity.position;
    entity.delta_pos_m = { 0.0f, 0.0f };
    entity.inv_x = false;
    entity.inv_y = false;
}


static void update_positions(AppState& state)
{
    auto n_threads = state.device.entities.n_elements;

    bool proc = cuda_no_errors("update_positions");
    assert(proc);

    gpu_update_positions<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(state.device.entities, n_threads);

    proc &= cuda_launch_success("gpu_update_positions");
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