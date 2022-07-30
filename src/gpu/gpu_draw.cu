#include "gpu_include.cuh"

#include <cassert>


constexpr auto N_SCREEN_PIXELS = SCREEN_HEIGHT_PX * SCREEN_WIDTH_PX;


constexpr auto N_PIXELS_PER_PLAYER = PLAYER_WIDTH_PX * PLAYER_HEIGHT_PX;
constexpr auto N_PIXELS_PER_BLUE = BLUE_WIDTH_PX * BLUE_HEIGHT_PX;
constexpr auto N_PIXELS_PER_WALL = WALL_WIDTH_PX * WALL_HEIGHT_PX;
constexpr auto N_ENTITY_PIXELS = 
    N_PLAYER_ENTITIES * N_PIXELS_PER_PLAYER +
    N_BLUE_ENTITIES * N_PIXELS_PER_BLUE +
    N_BROWN_ENTITIES * N_PIXELS_PER_WALL;


namespace gpuf
{
/********************************/

constexpr auto PLAYER_PIXELS_BEGIN = 0U;
constexpr auto PLAYER_PIXELS_END = PLAYER_PIXELS_BEGIN + N_PLAYER_ENTITIES * N_PIXELS_PER_PLAYER;
constexpr auto BLUE_PIXELS_BEGIN = PLAYER_PIXELS_END;
constexpr auto BLUE_PIXELS_END = BLUE_PIXELS_BEGIN + N_BLUE_ENTITIES * N_PIXELS_PER_BLUE;
constexpr auto BROWN_PIXELS_BEGIN = BLUE_PIXELS_END;
constexpr auto BROWN_PIXELS_END = BROWN_PIXELS_BEGIN + N_BROWN_ENTITIES * N_PIXELS_PER_WALL;


class EntityPixel
{
public:
    u32 pixel_id;
    u32 entity_id;
    u32 bitmap_offset;
};


GPU_FUNCTION
static EntityPixel get_entity_pixel(u32 entity_pixel_id)
{
    assert(entity_pixel_id < N_ENTITY_PIXELS);

    u32 p_begin = 0;
    u32 pixels_per_entity = 0;
    auto get_id = gpuf::player_id;

    if(gpuf::id_in_range(entity_pixel_id, PLAYER_PIXELS_BEGIN, PLAYER_PIXELS_END))
    {
        p_begin = PLAYER_PIXELS_BEGIN;
        pixels_per_entity = N_PIXELS_PER_PLAYER;
        get_id = gpuf::player_id;
    }
    else if(gpuf::id_in_range(entity_pixel_id, BLUE_PIXELS_BEGIN, BLUE_PIXELS_END))
    {
        p_begin = BLUE_PIXELS_BEGIN;
        pixels_per_entity = N_PIXELS_PER_BLUE;
        get_id = gpuf::blue_id;
    }
    else if(gpuf::id_in_range(entity_pixel_id, BROWN_PIXELS_BEGIN, BROWN_PIXELS_END))
    {
        p_begin = BROWN_PIXELS_BEGIN;
        pixels_per_entity = N_PIXELS_PER_WALL;
        get_id = gpuf::brown_id;        
    }
    else
    {
        assert(false);
        auto error = 0u - 1;
        return { error, error, error };
    }

    auto entity_pixel_offset = entity_pixel_id - p_begin;
    auto entity_offset = entity_pixel_offset / pixels_per_entity;
    auto entity_id = get_id(entity_offset);

    auto bitmap_offset = entity_pixel_offset - entity_offset * pixels_per_entity;

    return { entity_pixel_id, entity_id, bitmap_offset };
}


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
static void gpu_draw_entity_pixels(ScreenProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == N_ENTITY_PIXELS);

    auto& device = *props.device_p;

    auto entity_pixel_id = (u32)t;

    auto entity_info = gpuf::get_entity_pixel(entity_pixel_id);
    auto& entity = device.entities.data[entity_info.entity_id];

    if(!gpuf::is_drawable(entity))
    {
        return;
    }

    auto width_px = entity.bitmap.width;
    auto height_px = entity.bitmap.height;

    auto pixel_id = entity_info.bitmap_offset;

    auto& screen_dst = props.device_p->screen_pixels;

    auto entity_screen_pos_m = gpuf::sub_delta_m(entity.position, props.screen_pos);

    auto draw_rect_m = gpuf::get_screen_rect(entity, entity_screen_pos_m);

    auto pixel_height = entity.height_m / height_px;
    auto pixel_width = entity.width_m / width_px;

    auto pixel_y = pixel_id / width_px;
    auto pixel_x = pixel_id - pixel_y * width_px;    

    draw_rect_m.y_begin += pixel_y * pixel_height;
    draw_rect_m.y_end = draw_rect_m.y_begin + pixel_height;

    draw_rect_m.x_begin += pixel_x * pixel_width;
    draw_rect_m.x_end = draw_rect_m.x_begin + pixel_width;

    auto screen_rect_m = gpuf::make_rect(props.screen_width_m, props.screen_height_m);    

    auto is_offscreen = !gpuf::rect_intersect(draw_rect_m, screen_rect_m);
    if(is_offscreen)
    {
        return;
    }

    auto color = screen_dst.width / props.screen_width_m > 75 ? entity.bitmap.data[pixel_id] : entity.avg_color;

    gpuf::clamp_rect(draw_rect_m, screen_rect_m);
    auto rect_px = gpuf::to_pixel_rect(draw_rect_m, props.screen_width_m, screen_dst.width);

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
        bool result = cuda::no_errors("gpu::render");
        assert(result);
        
        constexpr auto tile_pixel_threads = N_SCREEN_PIXELS;
        constexpr auto tile_pixel_blocks = calc_thread_blocks(tile_pixel_threads);

        constexpr auto entity_pixel_threads = N_ENTITY_PIXELS;
        constexpr auto entity_pixel_blocks = calc_thread_blocks(entity_pixel_threads);


        auto props = make_screen_props(state);
        
        cuda_launch_kernel(gpu_draw_tiles, tile_pixel_blocks, THREADS_PER_BLOCK, props, tile_pixel_threads);
        result = cuda::launch_success("gpu_draw_tiles");
        assert(result); 

        cuda_launch_kernel(gpu_draw_entity_pixels, entity_pixel_blocks, THREADS_PER_BLOCK, props, entity_pixel_threads);
        result = cuda::launch_success("gpu_draw_entity_pixels");
        assert(result);
    }
}
