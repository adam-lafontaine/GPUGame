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
static void update_player_dt(Vec2Dr32& player_dt, InputRecord const& input)
{    
    player_dt = { 0.0f, 0.0f };

    if(input.input & INPUT::PLAYER_UP)
    {
        player_dt.y -= input.est_dt_frame;
    }

    if(input.input & INPUT::PLAYER_DOWN)
    {
        player_dt.y += input.est_dt_frame;
    }

    if(input.input & INPUT::PLAYER_LEFT)
    {
        player_dt.x -= input.est_dt_frame;
    }

    if(input.input & INPUT::PLAYER_RIGHT)
    {
        player_dt.x += input.est_dt_frame;
    }

    if(player_dt.x != 0.0f && player_dt.y != 0.0f)
    {
        player_dt.x *= 0.707107f;
        player_dt.y *= 0.707107f;
    }
}


GPU_FUNCTION 
void apply_current_input(PlayerProps& player, InputList const& inputs, u64 frame)
{
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

    update_player_dt(player.props.dt[player.id], last);
}


GPU_FUNCTION
static void stop_wall(PlayerProps& ent, WallProps const& other)
{   
    if(!gpuf::is_active(ent.props.status[ent.id]) || !gpuf::is_active(other.props.status[other.id]))
    {
        return;
    }

    auto& d_ent = ent.props.delta_pos_m[ent.id];
    if(!d_ent.x && !d_ent.y)
    {
        return;
    }

    auto ent_pos = ent.props.next_position[ent.id];
    auto ent_dim = ent.props.dim_m[ent.id];
    auto other_pos = other.props.position[other.id];
    auto other_dim = other.props.dim_m[other.id];

    auto delta = gpuf::sub_delta_m(ent_pos, other_pos);

    auto rect_ent = gpuf::make_rect(delta, ent_dim.x, ent_dim.y);
    auto rect_other = gpuf::make_rect(other_dim.x, other_dim.y);

    if(!gpuf::rect_intersect(rect_ent, rect_other))
    {
        return;
    }

    auto mm = 0.001f;

    if (d_ent.x > 0.0f && rect_ent.x_end >= rect_other.x_begin)
    {
        d_ent.x = rect_other.x_begin - 0.5f * mm;
    }
    else if (d_ent.x < 0.0f && rect_ent.x_begin <= rect_other.x_end)
    {
        d_ent.x = rect_other.x_end + 0.5f * mm;
    }

    if(d_ent.y > 0.0f && rect_ent.y_end >= rect_other.y_begin)
    {
        d_ent.y = rect_other.y_begin - 0.5f * mm;
    }
    else if (d_ent.y < 0.0f && rect_ent.y_begin <= rect_other.y_end)
    {
        d_ent.y = rect_other.y_end + 0.5f * mm;
    }
}


GPU_FUNCTION
static void bounce_wall(BlueProps& ent, WallProps const& other)
{
    if(!gpuf::is_active(ent.props.status[ent.id]) || !gpuf::is_active(other.props.status[other.id]))
    {
        return;
    }

    auto& d_ent = ent.props.delta_pos_m[ent.id];
    if(!d_ent.x && !d_ent.y)
    {
        return;
    }

    auto ent_pos = ent.props.next_position[ent.id];
    auto ent_dim = ent.props.dim_m[ent.id];
    auto other_pos = other.props.position[other.id];
    auto other_dim = other.props.dim_m[other.id];

    auto delta = gpuf::sub_delta_m(ent_pos, other_pos);

    auto rect_ent = gpuf::make_rect(delta, ent_dim.x, ent_dim.y);
    auto rect_other = gpuf::make_rect(other_dim.x, other_dim.y);

    if(!gpuf::rect_intersect(rect_ent, rect_other))
    {
        return;
    }

    auto mm = 0.001f;
    auto& dt = ent.props.dt[ent.id];

    if (d_ent.x > 0.0f && rect_ent.x_end >= rect_other.x_begin)
    {
        d_ent.x = rect_other.x_begin - 0.5f * mm;
        dt.x *= -1.0f;
    }
    else if (d_ent.x < 0.0f && rect_ent.x_begin <= rect_other.x_end)
    {
        d_ent.x = rect_other.x_end + 0.5f * mm;
        dt.x *= -1.0f;
    }

    if (d_ent.y > 0.0f && rect_ent.y_end >= rect_other.y_begin)
    {
        d_ent.y = rect_other.y_begin - 0.5f * mm;
        dt.y *= -1.0f;
    }
    else if (d_ent.y < 0.0f && rect_ent.y_begin <= rect_other.y_end)
    {
        d_ent.y = rect_other.y_end + 0.5f * mm;
        dt.y *= -1.0f;
    }
}


GPU_FUNCTION
static void blue_blue(BlueProps& ent, BlueProps const& other)
{
    if(!gpuf::is_active(ent.props.status[ent.id]) || !gpuf::is_active(other.props.status[other.id]))
    {
        return;
    }

    auto& d_ent = ent.props.delta_pos_m[ent.id];
    if(!d_ent.x && !d_ent.y)
    {
        return;
    }

    auto ent_pos = ent.props.next_position[ent.id];
    auto ent_dim = ent.props.dim_m[ent.id];
    auto other_pos = other.props.position[other.id];
    auto other_dim = other.props.dim_m[other.id];

    auto delta = gpuf::sub_delta_m(ent_pos, other_pos);

    auto rect_ent = gpuf::make_rect(delta, ent_dim.x, ent_dim.y);
    auto rect_other = gpuf::make_rect(other_dim.x, other_dim.y);

    if(!gpuf::rect_intersect(rect_ent, rect_other))
    {
        return;
    }

    auto mm = 0.001f;
    auto& dt = ent.props.dt[ent.id];

    if (d_ent.x > 0.0f && rect_ent.x_end >= rect_other.x_begin)
    {
        d_ent.x = rect_other.x_begin - 0.5f * mm;
        dt.x *= -1.0f;
    }
    else if (d_ent.x < 0.0f && rect_ent.x_begin <= rect_other.x_end)
    {
        d_ent.x = rect_other.x_end + 0.5f * mm;
        dt.x *= -1.0f;
    }

    if (d_ent.y > 0.0f && rect_ent.y_end >= rect_other.y_begin)
    {
        d_ent.y = rect_other.y_begin - 0.5f * mm;
        dt.y *= -1.0f;
    }
    else if (d_ent.y < 0.0f && rect_ent.y_begin <= rect_other.y_end)
    {
        d_ent.y = rect_other.y_end + 0.5f * mm;
        dt.y *= -1.0f;
    }
}


GPU_FUNCTION
static void blue_player(BlueProps& ent, PlayerProps const& other)
{
    if(!gpuf::is_active(ent.props.status[ent.id]) || !gpuf::is_active(other.props.status[other.id]))
    {
        return;
    }

    auto& d_ent = ent.props.delta_pos_m[ent.id];
    if(!d_ent.x && !d_ent.y)
    {
        return;
    }

    auto ent_pos = ent.props.next_position[ent.id];
    auto ent_dim = ent.props.dim_m[ent.id];
    auto other_pos = other.props.position[other.id];
    auto other_dim = other.props.dim_m[other.id];

    auto delta = gpuf::sub_delta_m(ent_pos, other_pos);

    auto rect_ent = gpuf::make_rect(delta, ent_dim.x, ent_dim.y);
    auto rect_other = gpuf::make_rect(other_dim.x, other_dim.y);

    if(!gpuf::rect_intersect(rect_ent, rect_other))
    {
        return;
    }
}


GPU_FUNCTION
static void next_postion(PlayerProps& ent)
{
    if(!gpuf::is_active(ent.props.status[ent.id]))
    {
        return;
    }

    auto& delta_pos = ent.props.delta_pos_m[ent.id];
    auto dt = ent.props.dt[ent.id];
    auto speed = ent.props.speed[ent.id];

    delta_pos = gpuf::vec_mul(dt, speed);

    auto& next_pos = ent.props.next_position[ent.id];
    auto pos = ent.props.position[ent.id];

    next_pos = gpuf::add_delta(pos, delta_pos);
}


GPU_FUNCTION
static void next_postion(BlueProps& ent)
{
    if(!gpuf::is_active(ent.props.status[ent.id]))
    {
        return;
    }

    auto& delta_pos = ent.props.delta_pos_m[ent.id];
    auto dt = ent.props.dt[ent.id];
    auto speed = ent.props.speed[ent.id];

    delta_pos = gpuf::vec_mul(dt, speed);

    auto& next_pos = ent.props.next_position[ent.id];
    auto pos = ent.props.position[ent.id];

    next_pos = gpuf::add_delta(pos, delta_pos);
}


GPU_FUNCTION
static void update_position(PlayerProps& ent)
{
    if(!gpuf::is_active(ent.props.status[ent.id]))
    {
        return;
    }

    auto& pos = ent.props.position[ent.id];
    auto& next_pos = ent.props.next_position[ent.id];
    auto& delta_pos = ent.props.delta_pos_m[ent.id];

    pos = gpuf::add_delta(pos, delta_pos);
    next_pos = pos;
    delta_pos = { 0.0f, 0.0f };
}


GPU_FUNCTION
static void update_position(BlueProps& ent)
{
    if(!gpuf::is_active(ent.props.status[ent.id]))
    {
        return;
    }

    auto& pos = ent.props.position[ent.id];
    auto& next_pos = ent.props.next_position[ent.id];
    auto& delta_pos = ent.props.delta_pos_m[ent.id];

    pos = gpuf::add_delta(pos, delta_pos);
    next_pos = pos;
    delta_pos = { 0.0f, 0.0f };
}


GPU_FUNCTION
static void update_on_screen(PlayerProps& ent, ScreenProps const& props)
{
    if(!gpuf::is_active(ent.props.status[ent.id]))
    {
        return;
    }

    auto pos = ent.props.position[ent.id];
    auto entity_screen_pos_m = gpuf::sub_delta_m(pos, props.screen_pos);

    auto dim = ent.props.dim_m[ent.id];
    auto ent_rect = gpuf::make_rect(entity_screen_pos_m, dim.x, dim.y);

    auto screen_rect = gpuf::make_rect(props.screen_width_m, props.screen_height_m);

    auto is_onscreen = gpuf::rect_intersect(ent_rect, screen_rect);
    auto& status = ent.props.status[ent.id];
    if(is_onscreen)
    {
        gpuf::set_onscreen(status);
    }
    else
    {
        gpuf::set_offscreen(status);
    }
}


GPU_FUNCTION
static void update_on_screen(BlueProps& ent, ScreenProps const& props)
{
    if(!gpuf::is_active(ent.props.status[ent.id]))
    {
        return;
    }

    auto pos = ent.props.position[ent.id];
    auto entity_screen_pos_m = gpuf::sub_delta_m(pos, props.screen_pos);

    auto dim = ent.props.dim_m[ent.id];
    auto ent_rect = gpuf::make_rect(entity_screen_pos_m, dim.x, dim.y);

    auto screen_rect = gpuf::make_rect(props.screen_width_m, props.screen_height_m);

    auto is_onscreen = gpuf::rect_intersect(ent_rect, screen_rect);
    auto& status = ent.props.status[ent.id];
    if(is_onscreen)
    {
        gpuf::set_onscreen(status);
    }
    else
    {
        gpuf::set_offscreen(status);
    }
}


GPU_FUNCTION
static void update_on_screen(WallProps& ent, ScreenProps const& props)
{
    if(!gpuf::is_active(ent.props.status[ent.id]))
    {
        return;
    }

    auto pos = ent.props.position[ent.id];
    auto entity_screen_pos_m = gpuf::sub_delta_m(pos, props.screen_pos);

    auto dim = ent.props.dim_m[ent.id];
    auto ent_rect = gpuf::make_rect(entity_screen_pos_m, dim.x, dim.y);

    auto screen_rect = gpuf::make_rect(props.screen_width_m, props.screen_height_m);

    auto is_onscreen = gpuf::rect_intersect(ent_rect, screen_rect);
    auto& status = ent.props.status[ent.id];
    if(is_onscreen)
    {
        gpuf::set_onscreen(status);
    }
    else
    {
        gpuf::set_offscreen(status);
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

    auto entity_id = (u32)t;       

    if(gpuf::is_player(entity_id))
    {
        PlayerProps player{};
        player.id = gpuf::to_player_id(entity_id);
        player.props = device.player_soa;

        if(player.id == unified.user_player_id)
        {
            gpuf::apply_current_input(player, unified.current_inputs, unified.frame_count);
        }
        else
        {
            // previous input
        }   

        gpuf::next_postion(player);     
    }
    else if (gpuf::is_blue(entity_id))
    {
        BlueProps blue{};
        blue.id = gpuf::to_blue_id(entity_id);
        blue.props = device.blue_soa;

        gpuf::next_postion(blue);
    }
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

    PlayerProps player{};
    player.id = player_offset;
    player.props = device.player_soa;

    WallProps wall{};
    wall.id = wall_offset;
    wall.props = device.wall_soa;

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

    BlueProps blue{};
    blue.id = blue_offset;
    blue.props = device.blue_soa;

    WallProps wall{};
    wall.id = wall_offset;
    wall.props = device.wall_soa;

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

    BlueProps blue{};
    blue.id = blue_offset;
    blue.props = device.blue_soa;

    PlayerProps player{};
    player.id = player_offset;
    player.props = device.player_soa;

    gpuf::blue_player(blue, player);
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

    BlueProps blue{};
    blue.id = a_offset;
    blue.props = device.blue_soa;

    BlueProps other{};
    blue.id = b_offset;
    blue.props = device.blue_soa;

    gpuf::blue_blue(blue, other);
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

    auto entity_id = (u32)t;

    if(gpuf::is_player(entity_id))
    {
        PlayerProps player{};
        player.id = gpuf::to_player_id(entity_id);
        player.props = device.player_soa;

        gpuf::update_position(player);
        gpuf::update_on_screen(player, props);
    }
    else if (gpuf::is_blue(entity_id))
    {
        BlueProps blue{};
        blue.id = gpuf::to_blue_id(entity_id);
        blue.props = device.blue_soa;

        gpuf::update_position(blue);
        gpuf::update_on_screen(blue, props);
    }
    else if (gpuf::is_wall(entity_id))
    {
        WallProps wall{};
        wall.id = gpuf::to_wall_id(entity_id);
        wall.props = device.wall_soa;

        gpuf::update_on_screen(wall, props);
    }
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