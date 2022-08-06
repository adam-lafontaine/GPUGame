#include "gpu_include.cuh"

#include <cassert>


namespace COUNT
{
    constexpr auto PLAYER_WALL_COLLISIONS = PLAYER_ENTITIES * WALL_ENTITIES;
    constexpr auto BLUE_WALL_COLLISIONS = BLUE_ENTITIES * WALL_ENTITIES;
    constexpr auto PLAYER_BLUE_COLLISIONS =  PLAYER_ENTITIES * BLUE_ENTITIES;
    constexpr auto BLUE_BLUE_COLLISIONS = BLUE_ENTITIES * BLUE_ENTITIES;
}


namespace gpuf
{
/*************************/



GPU_FUNCTION
static bool entity_will_intersect(Entity const& lhs, Entity const& rhs)
{
    auto delta_m = gpuf::sub_delta_m(lhs.next_position, rhs.next_position);
    
    auto rhs_rect = gpuf::make_rect(rhs.width_m, rhs.height_m);
    auto lhs_rect = gpuf::make_rect(delta_m, lhs.width_m, lhs.height_m);

    return gpuf::rect_intersect(lhs_rect, rhs_rect);
}


GPU_FUNCTION
static void move_player(Entity& entity, InputRecord const& input)
{    
    entity.dt = { 0.0f, 0.0f };

    if(input.input & INPUT::PLAYER_UP)
    {
        entity.dt.y -= input.est_dt_frame;
    }

    if(input.input & INPUT::PLAYER_DOWN)
    {
        entity.dt.y += input.est_dt_frame;
    }

    if(input.input & INPUT::PLAYER_LEFT)
    {
        entity.dt.x -= input.est_dt_frame;
    }

    if(input.input & INPUT::PLAYER_RIGHT)
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
void apply_current_input(Entity& entity, InputList const& inputs, u64 frame)
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
static void stop_wall(Entity& ent, Entity const& wall)
{   
    if(!gpuf::is_active(ent) || !gpuf::is_active(wall))
    {
        return;
    }

    auto delta = gpuf::sub_delta_m(ent.position, wall.position);
    
    auto w = gpuf::make_rect(wall.width_m, wall.height_m);
    auto e_start = gpuf::make_rect(delta, ent.width_m, ent.height_m);
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
    if(!gpuf::is_active(ent) || !gpuf::is_active(wall))
    {
        return;
    }

    if(ent.delta_pos_m.x == 0.0f && ent.delta_pos_m.y == 0.0f)
    {
        return;
    }

    auto delta = gpuf::sub_delta_m(ent.position, wall.position);
    
    auto w = gpuf::make_rect(wall.width_m, wall.height_m);
    auto e_start = gpuf::make_rect(delta, ent.width_m, ent.height_m);
    auto e_finish = gpuf::add_delta(e_start, ent.delta_pos_m);

    if(!gpuf::rect_intersect(e_finish, w))
    {
        return;
    }

    auto e_x_finish = gpuf::add_delta(e_start, { ent.delta_pos_m.x, 0.0f });
    if(gpuf::rect_intersect(e_x_finish, w))
    {
        gpuf::set_inv_x(ent);
    }

    auto e_y_finish = gpuf::add_delta(e_start, { 0.0f, ent.delta_pos_m.y });
    if(gpuf::rect_intersect(e_y_finish, w))
    {
        gpuf::set_inv_y(ent);
    }
    
}


GPU_FUNCTION
static void blue_blue(Entity& a, Entity const& b)
{
    if(!gpuf::is_active(a) || !gpuf::is_active(b))
    {
        return;
    }

    if(a.delta_pos_m.x == 0.0f && a.delta_pos_m.y == 0.0f)
    {
        return;
    }

    auto delta = gpuf::sub_delta_m(a.position, b.next_position);
    
    auto b_finish = gpuf::make_rect(b.width_m, b.height_m);
    auto a_start = gpuf::make_rect(delta, a.width_m, a.height_m);
    auto a_finish = gpuf::add_delta(a_start, a.delta_pos_m);

    if(!gpuf::rect_intersect(a_finish, b_finish))
    {
        return;
    }

    auto a_x_finish = gpuf::add_delta(a_start, { a.delta_pos_m.x, 0.0f });
    if(gpuf::rect_intersect(a_x_finish, b_finish))
    {
        gpuf::set_inv_x(a);
    }

    auto a_y_finish = gpuf::add_delta(a_start, { 0.0f, a.delta_pos_m.y });
    if(gpuf::rect_intersect(a_y_finish, b_finish))
    {
        gpuf::set_inv_y(a);
    }
}


GPU_FUNCTION
static void player_blue(Entity const& player, Entity& blue)
{   
    if(!gpuf::is_active(player) || !gpuf::is_active(blue) || !gpuf::entity_will_intersect(player, blue))
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
    gpuf::set_inactive(blue);
}


GPU_FUNCTION
static void entity_next_position(Entity& entity)
{
    if(!gpuf::is_active(entity))
    {
        return;
    }

    entity.delta_pos_m = gpuf::vec_mul(entity.dt, entity.speed);
    entity.next_position = gpuf::add_delta(entity.position, entity.delta_pos_m);
}


GPU_FUNCTION
static void update_entity_position(Entity& entity, ScreenProps const& props)
{
    if(gpuf::is_inv_x(entity))
    {
        entity.delta_pos_m.x = 0.0f;
        entity.dt.x *= -1.0f;
    }

    if(gpuf::is_inv_y(entity))
    {
        entity.delta_pos_m.y = 0.0f;
        entity.dt.y *= -1.0f;
    }

    entity.position = gpuf::add_delta(entity.position, entity.delta_pos_m);

    entity.next_position = entity.position;
    entity.delta_pos_m = { 0.0f, 0.0f };
    gpuf::unset_inv(entity);   
}


GPU_FUNCTION
static void update_entity_on_screen(Entity& entity, ScreenProps const& props)
{
    auto screen_width_m = props.screen_width_m;
    auto screen_height_m = props.screen_height_m;

    auto entity_screen_pos_m = gpuf::sub_delta_m(entity.position, props.screen_pos);
    auto entity_rect_m = gpuf::get_screen_rect(entity, entity_screen_pos_m);
    auto screen_rect_m = gpuf::make_rect(screen_width_m, screen_height_m);  

    auto is_onscreen = gpuf::rect_intersect(entity_rect_m, screen_rect_m);
    if(is_onscreen)
    {
        gpuf::set_onscreen(entity);
    }
    else
    {
        gpuf::set_offscreen(entity);
    }
}




/*************************/
}


GPU_KERNAL
static void gpu_next_movable_positions(DeviceMemory* device_p, UnifiedMemory* unified_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::MOVABLE_ENTITIES);

    auto& device = *device_p;
    auto& unified = *unified_p;

    auto offset = (u32)t;
    auto& entity = device.entities.data[offset];

    

    if(gpuf::is_player(entity.id))
    {
        if(entity.id == unified.user_player_entity_id)
        {
            gpuf::apply_current_input(entity, unified.current_inputs, unified.frame_count);
        }
        else
        {
            // previous input
        }        
    }
    
    gpuf::entity_next_position(entity);
}


GPU_KERNAL 
static void gpu_player_wall(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::PLAYER_WALL_COLLISIONS);

    auto& device = *device_p;

    auto offset = (u32)t;

    auto player_offset = offset / COUNT::WALL_ENTITIES;
    auto wall_offset = offset - player_offset * COUNT::WALL_ENTITIES;

    auto& player = device.player_entities.data[player_offset];
    auto& wall = device.wall_entities.data[wall_offset];

    gpuf::stop_wall(player, wall);
}


GPU_KERNAL
static void gpu_blue_wall(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::BLUE_WALL_COLLISIONS);

    auto& device = *device_p;

    auto offset = (u32)t;

    auto blue_offset = offset / COUNT::WALL_ENTITIES;
    auto wall_offset = offset - blue_offset * COUNT::WALL_ENTITIES;

    auto& blue = device.blue_entities.data[blue_offset];
    auto& wall = device.wall_entities.data[wall_offset];

    gpuf::bounce_wall(blue, wall);
}


GPU_KERNAL
static void gpu_player_blue(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::PLAYER_BLUE_COLLISIONS);

    auto& device = *device_p;

    auto offset = (u32)t;

    auto player_offset = offset / COUNT::BLUE_ENTITIES;
    auto blue_offset = offset - player_offset * COUNT::BLUE_ENTITIES;

    auto& blue = device.blue_entities.data[blue_offset];
    auto& player = device.player_entities.data[player_offset];

    gpuf::player_blue(player, blue);
}


GPU_KERNAL
static void gpu_blue_blue(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::BLUE_BLUE_COLLISIONS);

    auto& device = *device_p;

    auto offset = (u32)t;

    auto a_offset = offset / COUNT::BLUE_ENTITIES;
    auto b_offset = offset - a_offset * COUNT::BLUE_ENTITIES;

    if(a_offset == b_offset)
    {
        return;
    }

    auto& a = device.blue_entities.data[a_offset];
    auto& b = device.blue_entities.data[b_offset];

    gpuf::blue_blue(a, b);
}


GPU_KERNAL 
static void gpu_update_entity_positions(ScreenProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::ENTITIES);

    auto& device = *props.device_p;

    auto offset = (u32)t;

    auto& entity = device.entities.data[offset];
    if(!gpuf::is_active(entity))
    {
        return;
    }

    gpuf::update_entity_position(entity, props);
    gpuf::update_entity_on_screen(entity, props);
}


namespace gpu
{    
    void update(AppState& state)
    {        
        bool result = cuda::no_errors("gpu::update");
        assert(result);

        constexpr auto entity_threads = COUNT::ENTITIES;
        constexpr auto entity_blocks = calc_thread_blocks(entity_threads);
        
        constexpr auto movable_threads = COUNT::MOVABLE_ENTITIES;
        constexpr auto movable_blocks = calc_thread_blocks(movable_threads);

        constexpr auto player_wall_threads = COUNT::PLAYER_WALL_COLLISIONS;
        constexpr auto player_wall_blocks = calc_thread_blocks(player_wall_threads);

        constexpr auto blue_wall_threads = COUNT::BLUE_WALL_COLLISIONS;
        constexpr auto blue_wall_blocks = calc_thread_blocks(blue_wall_threads);

        constexpr auto player_blue_threads = COUNT::PLAYER_BLUE_COLLISIONS;
        constexpr auto player_blue_blocks = calc_thread_blocks(player_blue_threads);

        constexpr auto blue_blue_threads = COUNT::BLUE_BLUE_COLLISIONS;
        constexpr auto blue_blue_blocks = calc_thread_blocks(blue_blue_threads);

        auto device_p = state.device_buffer.data;
        auto unified_p = state.unified_buffer.data;
        
        cuda_launch_kernel(gpu_next_movable_positions, movable_blocks, THREADS_PER_BLOCK, device_p, unified_p, movable_threads);
        result = cuda::launch_success("gpu_next_movable_positions");
        assert(result);
        
        cuda_launch_kernel(gpu_player_wall, player_wall_blocks, THREADS_PER_BLOCK, device_p, player_wall_threads);
        result = cuda::launch_success("gpu_player_wall");
        assert(result);
        
        cuda_launch_kernel(gpu_blue_wall, blue_wall_blocks, THREADS_PER_BLOCK, device_p, blue_wall_threads);
        result = cuda::launch_success("gpu_blue_wall");
        assert(result);
        
        cuda_launch_kernel(gpu_player_blue, player_blue_blocks, THREADS_PER_BLOCK, device_p, player_blue_threads);
        result = cuda::launch_success("gpu_player_blue");
        assert(result);
        
        cuda_launch_kernel(gpu_blue_blue, blue_blue_blocks, THREADS_PER_BLOCK, device_p, blue_blue_threads);
        result = cuda::launch_success("gpu_blue_blue");
        assert(result);

        auto props = make_screen_props(state);

        cuda_launch_kernel(gpu_update_entity_positions, entity_blocks, THREADS_PER_BLOCK, props, entity_threads);
        result = cuda::launch_success("gpu_update_entity_positions");
        assert(result);
        
    }
}