#include "gpu_include.cuh"

#include <cassert>


class DrawProps
{
public:

    DeviceMemory* device_ptr;
    UnifiedMemory* unified_ptr;

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

    auto& screen_dst = props.unified_ptr->screen_pixels;

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
    auto& unified = *props.unified_ptr;

    auto& screen_dst = unified.screen_pixels;
    auto& tiles = device.tilemap;

    assert(n_threads == screen_dst.width * screen_dst.height);

    auto pixel_id = (u32)t;

    auto pixel_y_px = pixel_id / props.screen_width_px;
    auto pixel_x_px = pixel_id - pixel_y_px * props.screen_width_px;    

    auto pixel_y_m = gpuf::px_to_m(pixel_y_px, props.screen_width_m, props.screen_width_px);
    auto pixel_x_m = gpuf::px_to_m(pixel_x_px, props.screen_width_m, props.screen_width_px);

    auto pixel_pos = gpuf::add_delta(props.screen_pos, { pixel_x_m, pixel_y_m });

    auto tile_x = pixel_pos.tile.x;
    auto tile_y = pixel_pos.tile.y;

    auto black = gpuf::to_pixel(30, 30, 30);

    if(tile_x < 0 || tile_y < 0 || tile_x >= WORLD_WIDTH_TILE || tile_y >= WORLD_HEIGHT_TILE)
    {
        screen_dst.data[pixel_id] = black;
        return;
    }

    auto& tile =  tiles.data[tile_y * WORLD_WIDTH_TILE + tile_x];

    screen_dst.data[pixel_id] = gpuf::get_tile_color(tile, pixel_pos.offset_m, props.screen_width_m, props.screen_width_px);
}


GPU_KERNAL
static void gpu_draw_entities(DrawProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& device = *props.device_ptr;

    auto& entities = device.entities;

    assert(n_threads == entities.n_elements);

    auto entity_id = (u32)t;

    gpuf::draw_entity(entities.data[entity_id], props);


    if(gpuf::is_player_entity(entity_id))
    {
        gpuf::draw_entity(device.user_player, props);
    }
    else if(gpuf::is_blue_entity(entity_id))
    {
        auto offset = gpuf::get_blue_offset(entity_id);
        gpuf::draw_entity(device.blue_entities.data[offset], props);        
    }
    else if(gpuf::is_brown_entity(entity_id))
    {
        auto offset = gpuf::get_brown_offset(entity_id);
        gpuf::draw_entity(device.wall_entities.data[offset], props);
    }
}


namespace gpu
{
    void render(AppState& state)
    {
        u32 n_pixels = state.props.screen_width_px * state.props.screen_height_px;

        DrawProps props{};
        props.device_ptr = state.device;
        props.unified_ptr = state.unified;
        props.screen_width_px = state.props.screen_width_px;
        props.screen_width_m = state.props.screen_width_m;
        props.screen_pos = state.props.screen_position;

        bool result = cuda::no_errors("gpu::render");
        assert(result);

        auto n_threads = n_pixels;
        gpu_draw_tiles<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

        result = cuda::launch_success("gpu_draw_tiles");
        assert(result);

        n_threads = N_ENTITIES;
        gpu_draw_entities<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

        result = cuda::launch_success("gpu_draw_entities");
        assert(result);
    }
}
