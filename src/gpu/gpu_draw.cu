#include "gpu_include.cuh"

#include <cassert>


class DrawProps
{
public:
    DeviceMemory* device_p;
    UnifiedMemory* unified_p;

    WorldPosition screen_pos;
    u32 screen_width_px;
    r32 screen_width_m;
};


namespace gpuf
{
/********************************/


GPU_FUNCTION
static Pixel player_pixel(Point2Dr32 const& position_offset)
{
    return to_pixel(200, 0, 0);
}


GPU_FUNCTION
static Pixel blue_pixel(Point2Dr32 const& position_offset)
{
    return to_pixel(0, 0, 100);
}


GPU_FUNCTION
static Pixel wall_pixel(Point2Dr32 const& position_offset)
{
    return to_pixel(150, 75, 0);
}


GPU_FUNCTION
static Pixel black_pixel(Point2Dr32 const& position_offset)
{
    return to_pixel(30, 30, 30);
}


GPU_FUNCTION
static void draw_entity_offset(Entity const& entity, Pixel& dst, Point2Dr32 const& offset)
{
    if(gpuf::is_player(entity.id))
    {
        dst = player_pixel(offset);
    }
    else if(gpuf::is_blue(entity.id))
    {
        dst = blue_pixel(offset);
    }
    else if(gpuf::is_wall(entity.id))
    {
        dst = wall_pixel(offset);
    }
    else
    {
        dst = black_pixel(offset);
    }    
}


GPU_FUNCTION
static WorldPosition get_pixel_world_position(u32 pixel_id, DrawProps const& props)
{
    auto pixel_y_px = pixel_id / props.screen_width_px;
    auto pixel_x_px = pixel_id - pixel_y_px * props.screen_width_px;    

    auto pixel_y_m = gpuf::px_to_m(pixel_y_px, props.screen_width_m, props.screen_width_px);
    auto pixel_x_m = gpuf::px_to_m(pixel_x_px, props.screen_width_m, props.screen_width_px);

    return gpuf::add_delta(props.screen_pos, { pixel_x_m, pixel_y_m });
}

GPU_FUNCTION
static void draw_entity(Entity const& entity, DrawProps const& props)
{
    if(!entity.is_active)
    {
        return;
    }

    auto& screen_dst = props.device_p->screen_pixels;

    auto screen_width_px = screen_dst.width;
    auto screen_height_px = screen_dst.height;
    auto screen_width_m = props.screen_width_m;
    auto screen_height_m = gpuf::px_to_m(screen_height_px, props.screen_width_m, screen_width_px);

    auto entity_screen_pos_m = gpuf::sub_delta_m(entity.position, props.screen_pos);

    auto entity_rect_m = gpuf::get_screen_rect(entity, entity_screen_pos_m);

    auto screen_rect_m = gpuf::make_rect(screen_width_m, screen_height_m);
    
    auto is_offscreen = !gpuf::rect_intersect(entity_rect_m, screen_rect_m);

    if(is_offscreen)
    {
        return;
    }

    auto rect_px = gpuf::to_pixel_rect(entity_rect_m, screen_width_m, screen_width_px);
    Point2Dr32 offset = { 0.0f, 0.0f };

    for(i32 y = rect_px.y_begin; y < rect_px.y_end; ++y)
    {        
        if(y < 0 || y >= screen_height_px)
        {
            continue;
        }

        offset.y = (r32)(y - rect_px.y_begin) / (rect_px.y_end - rect_px.y_begin);

        auto dst_row = screen_dst.data + y * screen_width_px;

        for(i32 x = rect_px.x_begin; x < rect_px.x_end; ++x)
        {
            if(x < 0 || x >= screen_width_px)
            {
                continue;
            }

            offset.x = (r32)(x - rect_px.x_begin) / (rect_px.x_end - rect_px.x_begin);    

            gpuf::draw_entity_offset(entity, dst_row[x], offset);
        }
    }    
}






GPU_FUNCTION
static Pixel get_tile_pixel(Tile const& tile, WorldPosition const& pixel_world_pos, DrawProps const& props)
{
    auto bitmap_w_px = tile.width;
    auto bitmap_w_m = TILE_LENGTH_M;
    auto pixel_bitmap_offset_m = pixel_world_pos.offset_m;

    auto bitmap_pixel_m = bitmap_w_m / bitmap_w_px;
    auto screen_pixel_m = props.screen_width_m / props.screen_width_px;

    if(screen_pixel_m > bitmap_pixel_m)
    {
        return *tile.avg_color;
    }

    auto offset_x_px = gpuf::floor_r32_to_i32(pixel_bitmap_offset_m.x * bitmap_w_px / bitmap_w_m);
    auto offset_y_px = gpuf::floor_r32_to_i32(pixel_bitmap_offset_m.y * bitmap_w_px / bitmap_w_m);

    auto bitmap_px_id = offset_y_px * bitmap_w_px + offset_x_px;

    assert(bitmap_px_id < tile.width * tile.height);
    
    return tile.bitmap_data[bitmap_px_id];
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

    auto& device = *props.device_p;

    auto& screen_dst = device.screen_pixels;
    auto& tiles = device.tilemap;

    assert(n_threads == screen_dst.width * screen_dst.height);

    auto pixel_id = (u32)t;
    auto& pixel_dst = screen_dst.data[pixel_id];

    auto pixel_world_pos = gpuf::get_pixel_world_position(pixel_id, props);

    int tile_x = pixel_world_pos.tile.x;
    int tile_y = pixel_world_pos.tile.y;    

    auto black = gpuf::to_pixel(30, 30, 30);    

    if(tile_x < 0 || tile_y < 0 || tile_x >= WORLD_WIDTH_TILE || tile_y >= WORLD_HEIGHT_TILE)
    {
        pixel_dst = black;
        return;
    }

    auto tile_id = tile_y * WORLD_WIDTH_TILE + tile_x;
    auto& tile = tiles.data[tile_id];

    pixel_dst = gpuf::get_tile_pixel(tile, pixel_world_pos, props);
}


GPU_KERNAL
static void gpu_draw_entities(DrawProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == N_ENTITIES);

    auto& device = *props.device_p;    

    auto offset = (u32)t;
    auto& entity = device.entities.data[offset];

    if(!entity.is_active)
    {
        return;
    }

    auto& screen_dst = props.device_p->screen_pixels;

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

    gpuf::draw_entity(entity, props);
}


namespace gpu
{
    void render(AppState& state)
    {
        u32 n_pixels = state.app_input.screen_width_px * state.app_input.screen_height_px;

        DrawProps props{};
        props.device_p = state.device_buffer.data;
        props.unified_p = state.unified_buffer.data;
        props.screen_width_px = state.app_input.screen_width_px;
        props.screen_width_m = state.app_input.screen_width_m;
        props.screen_pos = state.app_input.screen_position;

        bool result = cuda::no_errors("gpu::render");
        assert(result);
        
        auto tile_threads = n_pixels;
        auto tile_blocks = calc_thread_blocks(tile_threads);

        constexpr auto entity_threads = N_ENTITIES;
        constexpr auto entity_blocks = calc_thread_blocks(entity_threads);
        
        cuda_launch_kernel(gpu_draw_tiles, tile_blocks, THREADS_PER_BLOCK, props, tile_threads);
        result = cuda::launch_success("gpu_draw_tiles");
        assert(result);        

        cuda_launch_kernel(gpu_draw_entities, entity_blocks, THREADS_PER_BLOCK, props, entity_threads);
        result = cuda::launch_success("gpu_draw_entities");
        assert(result);
    }
}
