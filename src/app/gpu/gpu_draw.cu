#include "gpu_include.cuh"

#include <cassert>


class DrawProps
{
public:

    DeviceMemoryOld* device_ptr;
    UnifiedMemoryOld* unified_ptr;

    UnifiedMemory* unified;

    WorldPosition screen_pos;
    u32 screen_width_px;
    r32 screen_width_m;
};


namespace gpuf
{
/********************************/



GPU_FUNCTION
Pixel get_tile_color(DeviceTile const& tile, Point2Dr32 const& offset_m, r32 screen_width_m, u32 screen_width_px)
{
    Pixel color{};

    auto bitmap_w_px = TILE_WIDTH_PX;
    auto bitmap_w_m = TILE_LENGTH_M;    

    auto offset_x_px = gpuf::floor_r32_to_i32(offset_m.x * bitmap_w_px / bitmap_w_m);
    auto offset_y_px = gpuf::floor_r32_to_i32(offset_m.y * bitmap_w_px / bitmap_w_m);

    auto bitmap_px_id = offset_y_px * bitmap_w_px + offset_x_px;

    assert(bitmap_px_id < TILE_WIDTH_PX * TILE_HEIGHT_PX);

    auto bitmap_pixel_m = TILE_LENGTH_M / TILE_WIDTH_PX;
    auto screen_pixel_m = screen_width_m / screen_width_px;

    if(screen_pixel_m > bitmap_pixel_m)
    {
        color = *tile.avg_color;
    }
    else
    {
        color = tile.bitmap_data[bitmap_px_id];
    }

    return color;
}


GPU_FUNCTION
static void draw_entity(Entity const& entity, DrawProps const& props)
{
    if(!entity.is_active)
    {
        return;
    }

    auto& screen_dst = props.unified->screen_pixels;

    auto screen_width_px = screen_dst.width;
    auto screen_height_px = screen_dst.height;
    auto screen_width_m = props.screen_width_m;
    auto screen_height_m = screen_height_px * props.screen_width_m / screen_width_px;

    auto entity_screen_pos_m = gpuf::sub_delta_m(entity.position, props.screen_pos);

    auto entity_rect_m = gpuf::get_screen_rect(entity, entity_screen_pos_m);

    auto screen_rect_m = gpuf::make_rect(screen_width_m, screen_height_m);
    
    auto is_offscreen = !gpuf::rect_intersect(entity_rect_m, screen_rect_m);

    if(is_offscreen)
    {
        return;
    }

    gpuf::clamp_rect(entity_rect_m, screen_rect_m);

    auto entity_rect_px = gpuf::to_pixel_rect(entity_rect_m, screen_width_m, screen_width_px);
    
    for(u32 y = entity_rect_px.y_begin; y < entity_rect_px.y_end; ++y)
    {
        auto row = screen_dst.data + y * screen_width_px;
        for(u32 x = entity_rect_px.x_begin; x < entity_rect_px.x_end; ++x)
        {
            row[x] = entity.color;
        }
    }
}



/*******************************/
}



GPU_KERNAL
static void gpu_draw_tiles(DrawProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *props.device_ptr;
    auto& unified = *props.unified;

    auto& screen_dst = unified.screen_pixels;
    auto& tiles = device.tilemap;

    assert(n_threads == screen_dst.width * screen_dst.height);

    auto pixel_id = (u32)t;

    auto pixel_y_px = pixel_id / props.screen_width_px;
    auto pixel_x_px = pixel_id - pixel_y_px * props.screen_width_px;    

    auto pixel_y_m = gpuf::px_to_m(pixel_y_px, props.screen_width_m, props.screen_width_px);
    auto pixel_x_m = gpuf::px_to_m(pixel_x_px, props.screen_width_m, props.screen_width_px);

    auto pixel_world_pos = gpuf::add_delta(props.screen_pos, { pixel_x_m, pixel_y_m });

    auto tile_x = pixel_world_pos.tile.x;
    auto tile_y = pixel_world_pos.tile.y;

    auto black = gpuf::to_pixel(30, 30, 30);

    if(tile_x < 0 || tile_y < 0 || tile_x >= WORLD_WIDTH_TILE || tile_y >= WORLD_HEIGHT_TILE)
    {
        screen_dst.data[pixel_id] = black;
        return;
    }

    auto& tile = tiles.data[tile_y * WORLD_WIDTH_TILE + tile_x];

    screen_dst.data[pixel_id] = gpuf::get_tile_color(tile, pixel_world_pos.offset_m, props.screen_width_m, props.screen_width_px);
}


GPU_KERNAL
static void gpu_draw_players(DrawProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *props.device_ptr;

    assert(n_threads == N_PLAYER_ENTITIES);

    gpuf::draw_entity(device.user_player, props);
}


GPU_KERNAL
static void gpu_draw_blue_entities(DrawProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *props.device_ptr;

    assert(n_threads == N_BLUE_ENTITIES);

    auto offset = (u32)t;
    gpuf::draw_entity(device.blue_entities.data[offset], props);
}


GPU_KERNAL
static void gpu_draw_wall_entities(DrawProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *props.device_ptr;

    assert(n_threads == N_BROWN_ENTITIES);

    auto offset = (u32)t;
    gpuf::draw_entity(device.wall_entities.data[offset], props);
}


namespace gpu
{
    void render(AppState& state)
    {
        u32 n_pixels = state.app_input.screen_width_px * state.app_input.screen_height_px;

        DrawProps props{};
        props.device_ptr = state.device_p;
        props.unified_ptr = state.unified_p;
        props.unified = state.unified.data;
        props.screen_width_px = state.app_input.screen_width_px;
        props.screen_width_m = state.app_input.screen_width_m;
        props.screen_pos = state.app_input.screen_position;

        bool result = cuda::no_errors("gpu::render");
        assert(result);

        constexpr auto player_threads = N_PLAYER_ENTITIES;
        constexpr auto player_blocks = calc_thread_blocks(player_threads);

        constexpr auto blue_threads = N_BLUE_ENTITIES;
        constexpr auto blue_blocks = calc_thread_blocks(blue_threads);

        constexpr auto wall_threads = N_BROWN_ENTITIES;
        constexpr auto wall_blocks = calc_thread_blocks(wall_threads);

        auto tile_threads = n_pixels;
        auto tile_blocks = calc_thread_blocks(tile_threads);
        
        cuda_launch_kernel(gpu_draw_tiles, tile_blocks, THREADS_PER_BLOCK, props, tile_threads);
        result = cuda::launch_success("gpu_draw_tiles");
        assert(result);
        
        cuda_launch_kernel(gpu_draw_players, player_blocks, THREADS_PER_BLOCK, props, player_threads);
        result = cuda::launch_success("gpu_draw_players");
        assert(result);
        
        cuda_launch_kernel(gpu_draw_blue_entities, blue_blocks, THREADS_PER_BLOCK, props, blue_threads);
        result = cuda::launch_success("gpu_draw_blue_entities");
        assert(result);
        
        cuda_launch_kernel(gpu_draw_wall_entities, wall_blocks, THREADS_PER_BLOCK, props, wall_threads);
        result = cuda::launch_success("gpu_draw_wall_entities");
        assert(result);
    }
}
