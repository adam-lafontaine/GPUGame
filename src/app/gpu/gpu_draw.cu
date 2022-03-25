#include "gpu_include.cuh"

#include <cassert>


class TileProps
{
public:

    DeviceTileMatrix tiles;
    DeviceImage screen_dst;

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


/*******************************/
}



GPU_KERNAL
static void gpu_draw_tiles(TileProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }   

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
        props.screen_dst.data[pixel_id] = black;
        return;
    }

    auto& tile =  props.tiles.data[tile_y * WORLD_WIDTH_TILE + tile_x];

    props.screen_dst.data[pixel_id] = gpuf::get_tile_color(tile, pixel_pos.offset_m, props.screen_width_m, props.screen_width_px);
}


static void draw_tiles(AppState& state)
{
    auto& dst = state.unified.screen_pixels;

    u32 width = dst.width;
    u32 height = dst.height;
    u32 n_pixels = width * height;

    auto n_threads = n_pixels;

    TileProps props{};
    props.tiles = state.device.tilemap;
    props.screen_dst = dst;
    props.screen_width_px = state.props.screen_width_px;
    props.screen_width_m = state.props.screen_width_m;
    props.screen_pos = state.props.screen_position;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_draw_tiles<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


class DrawEntityProps
{
public:
    DeviceArray<Entity> entities;

    DeviceImage screen_dst;
    r32 screen_width_m;

    WorldPosition screen_pos;
};


GPU_KERNAL
static void gpu_draw_entities(DrawEntityProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto entity_id = t;

    auto& entity = props.entities.data[entity_id];    

    if(!entity.is_active)
    {
        return;
    }

    auto screen_width_px = props.screen_dst.width;
    auto screen_height_px = props.screen_dst.height;
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
        auto row = props.screen_dst.data + y * screen_width_px;
        for(u32 x = entity_rect_px.x_begin; x < entity_rect_px.x_end; ++x)
        {
            row[x] = entity.color;
        }
    }

}


static void draw_entities(AppState& state)
{
    auto& dst = state.unified.screen_pixels;

    auto n_threads = state.device.entities.n_elements;


    DrawEntityProps props{};
    props.entities = state.device.entities;
    props.screen_dst = dst;
    props.screen_pos = state.props.screen_position;
    props.screen_width_m = state.props.screen_width_m;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_draw_entities<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


namespace gpu
{
    void render(AppState& state)
    {
        draw_tiles(state);
        draw_entities(state);
    }
}
