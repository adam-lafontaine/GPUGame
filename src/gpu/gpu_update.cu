#include "gpu_include.cuh"

#include <cassert>


constexpr auto N_PLAYER_WALL_COLLISIONS = N_PLAYER_ENTITIES * N_BROWN_ENTITIES;
constexpr auto N_BLUE_WALL_COLLISIONS = N_BLUE_ENTITIES * N_BROWN_ENTITIES;
constexpr auto N_PLAYER_BLUE_COLLISIONS =  N_PLAYER_ENTITIES * N_BLUE_ENTITIES;
constexpr auto N_BLUE_BLUE_COLLISIONS = N_BLUE_ENTITIES * N_BLUE_ENTITIES;


namespace gpuf
{
/*************************/



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


GPU_FUNCTION
static void entity_next_position(Entity& entity)
{
    if(!entity.is_active)
    {
        return;
    }

    entity.delta_pos_m = gpuf::vec_mul(entity.dt, entity.speed);
    entity.next_position = gpuf::add_delta(entity.position, entity.delta_pos_m);
}


GPU_FUNCTION
static void entity_update_position(Entity& entity)
{
    if(!entity.is_active)
    {
        return;
    }

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

    // TODO: set screen pixel flag

    entity.next_position = entity.position;
    entity.delta_pos_m = { 0.0f, 0.0f };
    entity.inv_x = false;
    entity.inv_y = false;
}




/*************************/
}


GPU_KERNAL
static void gpu_next_player_positions(DeviceMemory* device_ptr, UnifiedMemory* unified, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *device_ptr;

    assert(n_threads == N_PLAYER_ENTITIES);

    gpuf::apply_current_input(device.user_player, unified->current_inputs, unified->frame_count);

    gpuf::entity_next_position(device.user_player);
}


GPU_KERNAL
static void gpu_next_blue_positions(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *device_p;

    assert(n_threads == N_BLUE_ENTITIES);

    auto offset = (u32)t;
    gpuf::entity_next_position(device.blue_entities.data[offset]);
}


GPU_KERNAL 
static void gpu_player_wall(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == N_PLAYER_WALL_COLLISIONS);

    auto& device = *device_p;

    auto offset = (u32)t;

    auto player_offset = offset / N_BROWN_ENTITIES;
    auto wall_offset = offset - player_offset * N_BROWN_ENTITIES;

    auto& player = device.user_player;
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

    assert(n_threads == N_BLUE_WALL_COLLISIONS);

    auto& device = *device_p;

    auto offset = (u32)t;

    auto blue_offset = offset / N_BROWN_ENTITIES;
    auto wall_offset = offset - blue_offset * N_BROWN_ENTITIES;

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

    assert(n_threads == N_PLAYER_BLUE_COLLISIONS);

    auto& device = *device_p;

    auto offset = (u32)t;

    auto player_offset = offset / N_BLUE_ENTITIES;
    auto blue_offset = offset - player_offset * N_BLUE_ENTITIES;

    auto& blue = device.blue_entities.data[blue_offset];
    auto& player = device.user_player;

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

    assert(n_threads == N_BLUE_BLUE_COLLISIONS);

    auto& device = *device_p;

    auto offset = (u32)t;

    auto a_offset = offset / N_BLUE_ENTITIES;
    auto b_offset = offset - a_offset * N_BLUE_ENTITIES;

    if(a_offset == b_offset)
    {
        return;
    }

    auto& a = device.blue_entities.data[a_offset];
    auto& b = device.blue_entities.data[b_offset];

    gpuf::blue_blue(a, b);
}


GPU_KERNAL
static void gpu_update_player_positions(DeviceMemory* device_ptr, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *device_ptr;

    assert(n_threads == N_PLAYER_ENTITIES);

    gpuf::entity_update_position(device.user_player);    
}


GPU_KERNAL
static void gpu_update_blue_positions(DeviceMemory* device_p, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *device_p;

    assert(n_threads == N_BLUE_ENTITIES);

    auto offset = (u32)t;
    gpuf::entity_update_position(device.blue_entities.data[offset]);
}


namespace gpu
{    
    void update(AppState& state)
    {        
        bool result = cuda::no_errors("gpu::update");
        assert(result);

        constexpr auto player_threads = N_PLAYER_ENTITIES;
        constexpr auto player_blocks = calc_thread_blocks(player_threads);

        constexpr auto blue_threads = N_BLUE_ENTITIES;
        constexpr auto blue_blocks = calc_thread_blocks(blue_threads);

        constexpr auto player_wall_threads = N_PLAYER_WALL_COLLISIONS;
        constexpr auto player_wall_blocks = calc_thread_blocks(player_wall_threads);

        constexpr auto blue_wall_threads = N_BLUE_WALL_COLLISIONS;
        constexpr auto blue_wall_blocks = calc_thread_blocks(blue_wall_threads);

        constexpr auto player_blue_threads = N_PLAYER_BLUE_COLLISIONS;
        constexpr auto player_blue_blocks = calc_thread_blocks(player_blue_threads);

        constexpr auto blue_blue_threads = N_BLUE_BLUE_COLLISIONS;
        constexpr auto blue_blue_blocks = calc_thread_blocks(blue_blue_threads);

        auto device_p = state.device_buffer.data;
        auto unified_p = state.unified_buffer.data;
        
        cuda_launch_kernel(gpu_next_player_positions, player_blocks, THREADS_PER_BLOCK, device_p, unified_p, player_threads);
        result = cuda::launch_success("gpu_next_player_positions");
        assert(result);
        
        cuda_launch_kernel(gpu_next_blue_positions, blue_blocks, THREADS_PER_BLOCK, device_p, blue_threads);
        result = cuda::launch_success("gpu_next_blue_positions");
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
        
        cuda_launch_kernel(gpu_update_player_positions, player_blocks, THREADS_PER_BLOCK, device_p, player_threads);
        result = cuda::launch_success("gpu_update_player_positions");
        assert(result);
        
        cuda_launch_kernel(gpu_update_blue_positions, blue_blocks, THREADS_PER_BLOCK, device_p, blue_threads);
        result = cuda::launch_success("gpu_update_blue_positions");
        assert(result);
    }
}