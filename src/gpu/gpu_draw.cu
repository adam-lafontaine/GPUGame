#include "gpu_include.cuh"

#include <cassert>


namespace COUNT
{
    constexpr auto PIXELS_PER_PLAYER = PLAYER_WIDTH_PX * PLAYER_HEIGHT_PX;
    constexpr auto PIXELS_PER_BLUE = BLUE_WIDTH_PX * BLUE_HEIGHT_PX;
    constexpr auto PIXELS_PER_WALL = WALL_WIDTH_PX * WALL_HEIGHT_PX;

    constexpr auto PLAYER_PIXELS = PLAYER_ENTITIES * PIXELS_PER_PLAYER;
    constexpr auto BLUE_PIXELS = BLUE_ENTITIES * PIXELS_PER_BLUE;
    constexpr auto WALL_PIXELS = WALL_ENTITIES * PIXELS_PER_WALL;
/*
    constexpr auto ENTITY_PIXELS = 
        PLAYER_PIXELS +
        BLUE_PIXELS +
        WALL_PIXELS;
*/
}


namespace gpuf
{
/********************************/
/*
constexpr auto PLAYER_PIXELS_BEGIN = 0U;
constexpr auto PLAYER_PIXELS_END = PLAYER_PIXELS_BEGIN + COUNT::PLAYER_ENTITIES * COUNT::PIXELS_PER_PLAYER;
constexpr auto BLUE_PIXELS_BEGIN = PLAYER_PIXELS_END;
constexpr auto BLUE_PIXELS_END = BLUE_PIXELS_BEGIN + COUNT::BLUE_ENTITIES * COUNT::PIXELS_PER_BLUE;
constexpr auto WALL_PIXELS_BEGIN = BLUE_PIXELS_END;
constexpr auto WALL_PIXELS_END = WALL_PIXELS_BEGIN + COUNT::WALL_ENTITIES * COUNT::PIXELS_PER_WALL;
*/

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


GPU_FUNCTION
static void draw_bitmap_pixel(Image const& bitmap, u32 pixel_id, Vec2Dr32 const& dim_m, Point2Dr32 const& screen_pos_m, ScreenProps const& screen_props)
{
    auto bitmap_width_px = bitmap.width;
    auto bitmap_height_px = bitmap.height;

    auto bitmap_pixel_height_m = dim_m.y / bitmap_height_px;
    auto bitmap_pixel_width_m = dim_m.x / bitmap_width_px;

    auto bitmap_pixel_offset_y = pixel_id / bitmap_width_px;
    auto bitmap_pixel_offset_x = pixel_id - bitmap_pixel_offset_y * bitmap_width_px;

    Vec2Dr32 bitmap_pixel_offset_m{};
    bitmap_pixel_offset_m.x = bitmap_pixel_offset_x * bitmap_pixel_width_m;
    bitmap_pixel_offset_m.y = bitmap_pixel_offset_y * bitmap_pixel_height_m;

    auto bitmap_pixel_screen_pos_m = gpuf::add(screen_pos_m, bitmap_pixel_offset_m);

    auto draw_rect_m = gpuf::make_rect(bitmap_pixel_screen_pos_m, bitmap_pixel_width_m, bitmap_pixel_height_m);
    auto screen_rect_m = gpuf::make_rect(screen_props.screen_width_m, screen_props.screen_height_m);

    auto is_offscreen = !gpuf::rect_intersect(draw_rect_m, screen_rect_m);
    if(is_offscreen)
    {
        return;
    }

    auto screen_dst = screen_props.device_p->screen_pixels;
    auto color = bitmap.data[pixel_id]; //screen_dst.width / screen_props.screen_width_m > 75 ? bitmap.data[pixel_id] : ent_props.avg_color[id];

    gpuf::clamp_rect(draw_rect_m, screen_rect_m);
    auto rect_px = gpuf::to_pixel_rect(draw_rect_m, screen_props.screen_width_m, screen_dst.width);

    for(i32 y = rect_px.y_begin; y < rect_px.y_end; ++y)
    {
        auto dst_row = screen_dst.data + y * screen_dst.width;
        for(i32 x = rect_px.x_begin; x < rect_px.x_end; ++x)
        {
            dst_row[x] = color;
        }
    }
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
static void gpu_draw_player_pixels(ScreenProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::PLAYER_PIXELS);

    auto& device = *props.device_p;
    auto ent_props = device.player_soa;

    auto pixel_id = (u32)t;
    auto id = pixel_id / COUNT::PIXELS_PER_PLAYER;

    if (!gpuf::is_drawable(ent_props.status[id]))
    {
        return;
    }

    auto entity_screen_pos_m = gpuf::subtract_abs(ent_props.position[id], props.screen_pos);

    gpuf::draw_bitmap_pixel(ent_props.bitmap[id], pixel_id, ent_props.dim_m[id], entity_screen_pos_m, props);
}


GPU_KERNAL
static void gpu_draw_blue_pixels(ScreenProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::BLUE_PIXELS);

    auto& device = *props.device_p;
    auto ent_props = device.blue_soa;

    auto pixel_id = (u32)t;
    auto id = pixel_id / COUNT::PIXELS_PER_BLUE;

    if (!gpuf::is_drawable(ent_props.status[id]))
    {
        return;
    }

    auto entity_screen_pos_m = gpuf::subtract_abs(ent_props.position[id], props.screen_pos);

    gpuf::draw_bitmap_pixel(ent_props.bitmap[id], pixel_id, ent_props.dim_m[id], entity_screen_pos_m, props);
}


GPU_KERNAL
static void gpu_draw_wall_pixels(ScreenProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    assert(n_threads == COUNT::WALL_PIXELS);

    auto& device = *props.device_p;
    auto ent_props = device.wall_soa;

    auto pixel_id = (u32)t;
    auto id = pixel_id / COUNT::PIXELS_PER_WALL;

    if (!gpuf::is_drawable(ent_props.status[id]))
    {
        return;
    }

    auto entity_screen_pos_m = gpuf::subtract_abs(ent_props.position[id], props.screen_pos);

    gpuf::draw_bitmap_pixel(ent_props.bitmap[id], pixel_id, ent_props.dim_m[id], entity_screen_pos_m, props);
}


namespace gpu
{
    void render(AppState& state)
    {
        bool result = cuda::no_errors("gpu::render");
        assert(result);
        
        constexpr auto tile_pixel_threads = COUNT::SCREEN_PIXELS;
        constexpr auto tile_pixel_blocks = calc_thread_blocks(tile_pixel_threads);

        constexpr auto player_pixel_threads = COUNT::PLAYER_PIXELS;
        constexpr auto player_pixel_blocks = calc_thread_blocks(player_pixel_threads);

        constexpr auto blue_pixel_threads = COUNT::BLUE_PIXELS;
        constexpr auto blue_pixel_blocks = calc_thread_blocks(blue_pixel_threads);

        constexpr auto wall_pixel_threads = COUNT::WALL_PIXELS;
        constexpr auto wall_pixel_blocks = calc_thread_blocks(wall_pixel_threads);


        auto props = make_screen_props(state);
        
        cuda_launch_kernel(gpu_draw_tiles, tile_pixel_blocks, THREADS_PER_BLOCK, props, tile_pixel_threads);
        result = cuda::launch_success("gpu_draw_tiles");
        assert(result);

        cuda_launch_kernel(gpu_draw_player_pixels, player_pixel_blocks, THREADS_PER_BLOCK, props, player_pixel_threads);
        result = cuda::launch_success("gpu_draw_player_pixels");
        assert(result);

        cuda_launch_kernel(gpu_draw_blue_pixels, blue_pixel_blocks, THREADS_PER_BLOCK, props, blue_pixel_threads);
        result = cuda::launch_success("gpu_draw_blue_pixels");
        assert(result);

        cuda_launch_kernel(gpu_draw_wall_pixels, wall_pixel_blocks, THREADS_PER_BLOCK, props, wall_pixel_threads);
        result = cuda::launch_success("gpu_draw_wall_pixels");
        assert(result);
    }
}
