#include "render.hpp"
#include "../device/cuda_def.cuh"

#include <cassert>


constexpr int THREADS_PER_BLOCK = 1024;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


GPU_CONSTEXPR_FUNCTION
r32 pixel_distance_m(u32 n_pixels, r32 width_m, u32 width_px)
{
    auto dist = n_pixels * width_m / width_px;

    if(fabs(dist) < 0.0001)
    {
        return 0.0f;
    }

    return dist;
}


GPU_CONSTEXPR_FUNCTION
Pixel to_pixel(u8 red, u8 green, u8 blue)
{
    Pixel p{};
    p.alpha = 255;
    p.red = red;
    p.green = green;
    p.blue = blue;
}


GPU_FUNCTION
inline i32 cuda_floor_r32_to_i32(r32 value)
{
    return (i32)(floorf(value));
}


GPU_FUNCTION
static WorldPosition add_delta(WorldPosition const& pos, Vec2Dr32 const& delta)
{
    WorldPosition added{};

    r32 dist_m = pos.offset_m.x + delta.x;
    i32 delta_tile = cuda_floor_r32_to_i32(dist_m / TILE_LENGTH_M);
    
    added.tile.x = pos.tile.x + delta_tile;
    added.offset_m.x = dist_m - (r32)delta_tile * TILE_LENGTH_M;

    dist_m = pos.offset_m.y + delta.y;
    delta_tile = cuda_floor_r32_to_i32(dist_m / TILE_LENGTH_M);
    
    added.tile.y = pos.tile.y + delta_tile;
    added.offset_m.y = dist_m - (r32)delta_tile * TILE_LENGTH_M;

    return added; 
}


GPU_FUNCTION
static Vec2Dr32 subtract(WorldPosition const& lhs, WorldPosition const& rhs)
{
    Vec2Dr32 delta{};

    delta.x = TILE_LENGTH_M * (lhs.tile.x - rhs.tile.x) + lhs.offset_m.x - rhs.offset_m.x;
    delta.y = TILE_LENGTH_M * (lhs.tile.y - rhs.tile.y) + lhs.offset_m.y - rhs.offset_m.y;

    return delta;
}


class TileProps
{
public:

    DeviceTileMatrix tiles;
    DeviceImage screen_dst;

    WorldPosition screen_pos;
    u32 screen_width_px;
    r32 screen_width_m;
};


GPU_KERNAL
static void gpu_draw_tiles(TileProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto black = to_pixel(30, 30, 30);

    auto pixel_id = (u32)t;

    auto pixel_y_px = pixel_id / props.screen_width_px;
    auto pixel_x_px = pixel_id - pixel_y_px * props.screen_width_px;    

    auto pixel_y_m = pixel_distance_m(pixel_y_px, props.screen_width_m, props.screen_width_px);
    auto pixel_x_m = pixel_distance_m(pixel_x_px, props.screen_width_m, props.screen_width_px);

    auto pixel_pos = add_delta(props.screen_pos, { pixel_x_m, pixel_y_m });

    auto px = pixel_pos.tile.x;
    auto py = pixel_pos.tile.y;

    if(px < 0 || py < 0 || px >= WORLD_WIDTH_TILE || py >= WORLD_HEIGHT_TILE)
    {
        props.screen_dst.data[pixel_id] = black;
        return;
    }

    i32 tile_id = py * WORLD_WIDTH_TILE + px;
    
    props.screen_dst.data[pixel_id] = props.tiles.data[tile_id].color;
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


class EntityProps
{
public:
    DeviceArray<Entity> entities;
    DeviceImage screen_dst;

    WorldPosition screen_pos;

    u32 screen_width_px;
    u32 screen_height_px;

    r32 screen_width_m;
};


GPU_FUNCTION
Rect2Dr32 make_rect(r32 width, r32 height)
{
    Rect2Dr32 r{};

    r.x_begin = 0.0f;
    r.x_end = width;
    r.y_begin = 0.0f
    r.y_end = height;

    return r;
}


GPU_FUNCTION
Rect2Dr32 get_entity_rect(Entity const& entity, Point2Dr32 const& pos)
{
    Rect2Dr32 r{};

    // pos at bottom center of rect
    r.x_begin = pos.x - 0.5f * entity.width;
    r.x_begin = r.x_begin + entity.width;
    r.y_end = pos.y;
    r.y_begin = r.y_end - entity.height;

    return r;
}





GPU_KERNAL
static void gpu_draw_entities(EntityProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto entity_id = (u32)t;

    auto& entity = props.entities.data[entity_id];

    auto screen_pos_m = subtract(entity.position, props.screen_pos);
    auto screen_height_m = props.screen_width_m * props.screen_height_px / props.screen_width_px;

    auto entity_m = get_entity_rect(entity, screen_pos_m);
    auto screen_m = make_rect(props.screen_width_m, screen_height_m);

    
    auto is_offscreen = 
        entity_rect.x_end < 0.0f ||
        entity_rect.x_begin > props.screen_width_m ||
        entity_rect.y_end < 0.0f ||
        entity_rect.y_begin > screen_height_m; 

    if(is_offscreen)
    {
        return;
    }








}




/*
static void draw_entities(AppState& state)
{
    auto& dst = state.unified.screen_pixels;

    auto n_threads = state.device.entities.n_elements;


    EntityProps props{};
    props.entities = state.device.entities;
    props.screen_dst = dst;


    bool proc = cuda_no_errors();
    assert(proc);

    gpu_draw_entities<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}
*/

void render(AppState& state)
{    
    draw_tiles(state);

}


GPU_KERNAL
void gpu_init_tiles(DeviceTileMatrix tiles, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto tile_id = (u32)t;

    assert(tile_id < tiles.width * tiles.height);

    auto tile_y = tile_id / tiles.width;
    auto tile_x = tile_id - tile_y * tiles.width;

    auto& tile = tiles.data[tile_id];
    tile.color.alpha = 255;

    if(tile_y == 0 && tile_x == 0)
    {
        tile.color.red = 255;
        tile.color.green = 0;
        tile.color.blue = 0;

        return;
    }

    if((tile_y % 2 == 0 && tile_x % 2 != 0) || (tile_y % 2 != 0 && tile_x % 2 == 0))
    {
        tile.color.red = 255;
        tile.color.green = 255;
        tile.color.blue = 255;
    }
    else
    {
        tile.color.red = (u8)(tile_y / 2);
        tile.color.green = (u8)((tiles.width - tile_x) / 2);
        tile.color.blue = (u8)((tiles.height - tile_y) / 2);
    }

}


GPU_FUNCTION
void init_player(Entity& player)
{
    player.width = 0.5f;
    player.height = 0.6f;

    player.color.alpha = 255;
    player.color.red = 0;
    player.color.green = 0;
    player.color.blue = 0;

    player.position.tile = { 5, 5 };
    player.position.offset_m = { 0.2f, 0.2f };
}


GPU_KERNAL
void gpu_init_entities(DeviceArray<Entity> entities, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto entity_id = (u32)t;

    if(entity_id == PLAYER_ID)
    {
        init_player(entities.data[entity_id]);
    }
}


void init_device_memory(DeviceMemory const& device)
{
    bool proc = cuda_no_errors();
    assert(proc);

    auto n_threads = device.tilemap.width * device.tilemap.height;
    gpu_init_tiles<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(device.tilemap, n_threads);

    proc &= cuda_launch_success();
    assert(proc);

    n_threads = device.entities.n_elements;
    gpu_init_entities<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(device.entities, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}