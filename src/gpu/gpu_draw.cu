#include "gpu_include.cuh"

#include <cassert>

constexpr auto N_SECTIONS_PER_ENTITY = 4;
constexpr auto N_ENTITY_SECTIONS = N_ENTITIES * N_SECTIONS_PER_ENTITY;


namespace gpuf
{
/********************************/


GPU_FUNCTION
static WorldPosition get_pixel_world_position(u32 pixel_id, ScreenProps const& props)
{
    auto& screen_dst = props.device_p->screen_pixels;

    auto screen_width_px = screen_dst.width;

    auto pixel_y_px = pixel_id / screen_width_px;
    auto pixel_x_px = pixel_id - pixel_y_px * screen_width_px;    

    auto pixel_y_m = gpuf::px_to_m(pixel_y_px, props.screen_width_m, screen_width_px);
    auto pixel_x_m = gpuf::px_to_m(pixel_x_px, props.screen_width_m, screen_width_px);

    return gpuf::add_delta(props.screen_pos, { pixel_x_m, pixel_y_m });
}


GPU_FUNCTION
static Pixel get_tile_pixel(Tile const& tile, WorldPosition const& pixel_world_pos, ScreenProps const& props)
{
    auto bitmap_w_px = tile.width;
    auto bitmap_w_m = TILE_LENGTH_M;
    auto pixel_bitmap_offset_m = pixel_world_pos.offset_m;

    auto bitmap_pixel_m = bitmap_w_m / bitmap_w_px;
    auto screen_pixel_m = props.screen_width_m / props.device_p->screen_pixels.width;

    if(screen_pixel_m > bitmap_pixel_m)
    {
        return *tile.avg_color;
    }

    auto offset_x_px = gpuf::m_to_px(pixel_bitmap_offset_m.x, bitmap_w_m, bitmap_w_px);
    auto offset_y_px = gpuf::m_to_px(pixel_bitmap_offset_m.y, bitmap_w_m, bitmap_w_px);

    auto bitmap_px_id = offset_y_px * bitmap_w_px + offset_x_px;

    assert(bitmap_px_id < tile.width * tile.height);
    
    return tile.bitmap_data[bitmap_px_id];
}



/*******************************/
}


GPU_KERNAL
static void gpu_draw_tiles(ScreenProps props, u32 n_threads)
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
static void gpu_draw_entity_sections(ScreenProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == N_ENTITY_SECTIONS);

    auto& device = *props.device_p;    

    auto entity_section_id = (u32)t;
    auto entity_id = entity_section_id / N_SECTIONS_PER_ENTITY;
    auto& entity = device.entities.data[entity_id];

    if(!entity.is_active || entity.is_offscreen)
    {
        return;
    }

    auto section_id = entity_section_id - entity_id * N_SECTIONS_PER_ENTITY;

    auto& screen_dst = props.device_p->screen_pixels;

    auto entity_screen_pos_m = gpuf::sub_delta_m(entity.position, props.screen_pos);

    auto entity_rect_m = gpuf::get_screen_rect(entity, entity_screen_pos_m);

    auto section_height = entity.height_m / N_SECTIONS_PER_ENTITY;

    entity_rect_m.y_begin += section_id * entity.height_m / N_SECTIONS_PER_ENTITY;
    entity_rect_m.y_end = entity_rect_m.y_begin + section_height;

    auto screen_rect_m = gpuf::make_rect(props.screen_width_m, props.screen_height_m);    

    auto is_offscreen = !gpuf::rect_intersect(entity_rect_m, screen_rect_m);
    if(is_offscreen)
    {
        return;
    }

    auto color = section_id == N_SECTIONS_PER_ENTITY / 2 ? gpuf::to_pixel(100, 100, 100) : entity.color;

    gpuf::clamp_rect(entity_rect_m, screen_rect_m);
    auto rect_px = gpuf::to_pixel_rect(entity_rect_m, props.screen_width_m, screen_dst.width);

    for(i32 y = rect_px.y_begin; y < rect_px.y_end; ++y)
    {
        auto dst_row = screen_dst.data + y * screen_dst.width;
        for(i32 x = rect_px.x_begin; x < rect_px.x_end; ++x)
        {
            dst_row[x] = color;
        }
    }
}


namespace gpu
{
    void render(AppState& state)
    {
        u32 n_pixels = state.screen_pixels.width * state.screen_pixels.height;

        ScreenProps props{};
        props.device_p = state.device_buffer.data;
        props.screen_width_m = state.app_input.screen_width_m;
        props.screen_height_m = props.screen_width_m * state.screen_pixels.height / state.screen_pixels.width;
        props.screen_pos = state.app_input.screen_position;

        bool result = cuda::no_errors("gpu::render");
        assert(result);
        
        auto tile_threads = n_pixels;
        auto tile_blocks = calc_thread_blocks(tile_threads);

        constexpr auto entity_section_threads = N_ENTITY_SECTIONS;
        constexpr auto entity_section_blocks = calc_thread_blocks(entity_section_threads);
        
        cuda_launch_kernel(gpu_draw_tiles, tile_blocks, THREADS_PER_BLOCK, props, tile_threads);
        result = cuda::launch_success("gpu_draw_tiles");
        assert(result); 

        cuda_launch_kernel(gpu_draw_entity_sections, entity_section_blocks, THREADS_PER_BLOCK, props, entity_section_threads);
        result = cuda::launch_success("gpu_draw_entities");
        assert(result);
    }
}
