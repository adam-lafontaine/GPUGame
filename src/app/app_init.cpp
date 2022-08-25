#include "app_include.hpp"
#include "../libimage/libimage.hpp"

#include <cassert>

namespace img = libimage;


constexpr auto GRASS_TILE_PATH = "/home/adam/Repos/GPUGame/assets/tiles/basic_grass.png";
/*
namespace SIZE
{
    constexpr size_t Device_r32 = Entity_r32 * N_ENTITIES;
    constexpr size_t Device_Image = Entity_Image * N_ENTITIES;
    constexpr size_t Device_Pixel = Entity_Pixel * N_ENTITIES;
    constexpr size_t Device_WorldPosition = Entity_WorldPosition * N_ENTITIES;
    constexpr size_t Device_Vec2Dr32 = Entity_Vec2Dr32 * N_ENTITIES;
    
    constexpr size_t Unified_uStatus = Entity_uStatus * N_ENTITIES;
}
*/

static Pixel get_avg_color(Image const& image)
{
    auto sub_h = image.height;
    auto sub_w = image.width;

    u32 r = 0;
    u32 g = 0;
    u32 b = 0;
    for(u32 y = 0; y < sub_h; ++y)
    {
        for(u32 x = 0; x < sub_w; ++x)
        {
            auto p = image.data[y * image.width + x];
            r += p.red;
            g += p.green;
            b += p.blue;
        }
    }

    auto div = sub_h * sub_w;
    r /= div;
    g /= div;
    b /= div;

    return to_pixel((u8)r, (u8)g, (u8)b);
}


template <u32 W, u32 H>
static bool init_bitmap(MemoryBuffer<Pixel>& buffer, Image const& host_image, Bitmap<W, H>& bitmap)
{
    assert(host_image.width == bitmap.width);
    assert(host_image.height == bitmap.height);

    auto const n_pixels = host_image.width * host_image.height;

    bitmap.bitmap_data = cuda::push_elements(buffer, n_pixels);
    if(!bitmap.bitmap_data)
    {
        print_error("bitmap bitmap_data");
        return false;
    }

    if(!cuda::memcpy_to_device(host_image.data, bitmap.bitmap_data, n_pixels * sizeof(Pixel)))
    {
        print_error("bitmap memcpy");
        return false;
    }

    auto avg_image_color = get_avg_color(host_image);    

    bitmap.avg_color = cuda::push_elements(buffer, 1);
    if(!bitmap.avg_color)
    {
        print_error("bitmap avg_color");
        return false;
    }

    if(!cuda::memcpy_to_device(&avg_image_color, bitmap.avg_color, sizeof(Pixel)))
    {
        print_error("avg_color memcpy");
        return false;
    }

    return true;
}


static void fill_color(Image const& img, Pixel color)
{
    auto black = to_pixel(30, 30, 30);
    auto white = to_pixel(200, 200, 200);

    auto width = img.width;
    auto height = img.height;

    for(u32 y = 0; y < height; ++y)
    {
        auto dst_row = img.data + y * width;
        for(u32 x = 0; x < width; ++x)
        {
            if(y == height / 2 && x != 0 && x != width - 1)
            {
                dst_row[x] = white;
            }
            else if(y == height / 2 - 1 && x != 0 && x != width - 1)
            {
                dst_row[x] = black;
            }
            else if(y == height / 2 + 1 && x != 0 && x != width - 1)
            {
                dst_row[x] = black;
            }
            else
            {
                dst_row[x] = color;
            }            
        }
    }
}


static bool load_device_assets(MemoryBuffer<Pixel>& buffer, DeviceAssets& assets)
{
    Image read_img{};
    Image tile_img{};
    tile_img.width = Tile::width;
    tile_img.height = Tile::height;

    img::read_image_from_file(GRASS_TILE_PATH, read_img);
    img::resize_image(read_img, tile_img);

    Image player_img{};
    img::make_image(player_img, PlayerBitmap::width, PlayerBitmap::height);

    Image blue_img{};
    img::make_image(blue_img, BlueBitmap::width, BlueBitmap::height);

    Image wall_img{};
    img::make_image(wall_img, WallBitmap::width, WallBitmap::height);

    auto const cleanup = [&]()
    {
        img::destroy_image(read_img);
        img::destroy_image(tile_img);
        img::destroy_image(player_img);
        img::destroy_image(blue_img);
        img::destroy_image(wall_img);
    };
    
    // grass
    if(!init_bitmap(buffer, tile_img, assets.grass_tile))
    {
        cleanup();
        print_error("init grass");
        return false;
    }

    // temp make black
    auto black = to_pixel(0, 0, 0);
    fill_color(tile_img, black);
    if(!init_bitmap(buffer, tile_img, assets.black_tile))
    {
        cleanup();
        print_error("init black");
        return false;
    }

    // player
    auto red = to_pixel(200, 0, 0);
    fill_color(player_img, red);
    if(!init_bitmap(buffer, player_img, assets.player_bitmap))
    {
        cleanup();
        print_error("init player bitmap");
        return false;
    }

    // blue
    auto blue = to_pixel(0, 0, 100);
    fill_color(blue_img, blue);
    if(!init_bitmap(buffer, blue_img, assets.blue_bitmap))
    {
        cleanup();
        print_error("init blue bitmap");
        return false;
    }

    // wall
    auto brown = to_pixel(150, 75, 0);
    fill_color(wall_img, brown);
    if(!init_bitmap(buffer, wall_img, assets.wall_bitmap))
    {
        cleanup();
        print_error("init wall bitmap");
        return false;
    }

    cleanup();

    return true;
}


static bool init_image(Image& image, MemoryBuffer<Pixel>& buffer, u32 width, u32 height)
{
    auto const n_pixels = width * height;
    image.data = cuda::push_elements(buffer, n_pixels);
    if(!image.data)
    {
        return false;
    }

    image.width = width;
    image.height = height;

    return true;
}


static bool init_input_list(InputList& list, MemoryBuffer<InputRecord>& buffer)
{
    auto const n_records = INPUT::MAX_RECORDS;
    list.data = cuda::push_elements(buffer, n_records);
    if(!list.data)
    {
        return false;
    }

    list.capacity = n_records;
    list.size = 0;
    list.read_index = 0;

    return true;
}


static bool allocate_entity_status(AppState& state, DeviceMemory& device)
{
    auto& buffer = state.device_entity_ustatus_buffer;

    if(!cuda::device_malloc(state.device_entity_ustatus_buffer, SIZE::Entity_uStatus))
    {
        print_error("entity uStatus");
        return false;
    }

    auto status_data = cuda::push_elements(buffer, COUNT::PLAYER_ENTITIES);
    if(!status_data)
    {
        print_error("player status data");
        return false;
    }
    device.player_soa.status = status_data;

    status_data = cuda::push_elements(buffer, COUNT::BLUE_ENTITIES);
    if(!status_data)
    {
        print_error("blue status data");
        return false;
    }
    device.blue_soa.status = status_data;

    status_data = cuda::push_elements(buffer, COUNT::WALL_ENTITIES);
    if(!status_data)
    {
        print_error("wall status data");
        return false;
    }
    device.wall_soa.status = status_data;

    return true;
}


static bool allocate_entity_image(AppState& state, DeviceMemory& device)
{
    auto& buffer = state.device_entity_image_buffer;

    if(!cuda::device_malloc(buffer, SIZE::Entity_Image))
    {
        print_error("entity Image");
        return false;
    }

    auto image_data = cuda::push_elements(buffer, COUNT::PLAYER_ENTITIES);
    if(!image_data)
    {
        print_error("player image data");
        return false;
    }
    device.player_soa.bitmap = image_data;

    image_data = cuda::push_elements(buffer, COUNT::BLUE_ENTITIES);
    if(!image_data)
    {
        print_error("blue image data");
        return false;
    }
    device.blue_soa.bitmap = image_data;

    image_data = cuda::push_elements(buffer, COUNT::WALL_ENTITIES);
    if(!image_data)
    {
        print_error("wall image data");
        return false;
    }
    device.wall_soa.bitmap = image_data;

    return true;
}


static bool allocate_entity_pixel(AppState& state, DeviceMemory& device)
{
    auto& buffer = state.device_entity_pixel_buffer;

    if(!cuda::device_malloc(buffer, SIZE::Entity_Pixel))
    {
        print_error("entity Pixel");
        return false;
    }

    auto pixel_data = cuda::push_elements(buffer, COUNT::PLAYER_ENTITIES);
    if(!pixel_data)
    {
        print_error("player pixel data");
        return false;
    }
    device.player_soa.avg_color = pixel_data;

    pixel_data = cuda::push_elements(buffer, COUNT::BLUE_ENTITIES);
    if(!pixel_data)
    {
        print_error("blue pixel data");
        return false;
    }
    device.blue_soa.avg_color = pixel_data;

    pixel_data = cuda::push_elements(buffer, COUNT::WALL_ENTITIES);
    if(!pixel_data)
    {
        print_error("wall pixel data");
        return false;
    }
    device.wall_soa.avg_color = pixel_data;

    return true;
}


static bool allocate_entity_r32(AppState& state, DeviceMemory& device)
{
    auto& buffer = state.device_entity_r32_buffer;

    if(!cuda::device_malloc(buffer, SIZE::Entity_r32))
    {
        print_error("entity r32");
        return false;
    }

    auto r32_data = cuda::push_elements(buffer, COUNT::PLAYER_ENTITIES);
    if(!r32_data)
    {
        print_error("player speed data");
        return false;
    }
    device.player_soa.speed = r32_data;

    r32_data = cuda::push_elements(buffer, COUNT::BLUE_ENTITIES);
    if(!r32_data)
    {
        print_error("blue speed data");
        return false;
    }
    device.blue_soa.speed = r32_data;

    return true;
}


static bool allocate_entity_world_position(AppState& state, DeviceMemory& device)
{
    auto& buffer = state.device_entity_world_position_buffer;

    if(!cuda::device_malloc(buffer, SIZE::Entity_WorldPosition))
    {
        print_error("entity WorldPosition");
        return false;
    }

    auto pos_data = cuda::push_elements(buffer, COUNT::PLAYER_ENTITIES);
    if(!pos_data)
    {
        print_error("player position data");
        return false;
    }
    device.player_soa.position = pos_data;

    pos_data = cuda::push_elements(buffer, COUNT::PLAYER_ENTITIES);
    if(!pos_data)
    {
        print_error("player next position data");
        return false;
    }
    device.player_soa.next_position = pos_data;

    pos_data = cuda::push_elements(buffer, COUNT::BLUE_ENTITIES);
    if(!pos_data)
    {
        print_error("blue position data");
        return false;
    }
    device.blue_soa.position = pos_data;

    pos_data = cuda::push_elements(buffer, COUNT::BLUE_ENTITIES);
    if(!pos_data)
    {
        print_error("blue next position data");
        return false;
    }
    device.blue_soa.next_position = pos_data;

    pos_data = cuda::push_elements(buffer, COUNT::WALL_ENTITIES);
    if(!pos_data)
    {
        print_error("wall position data");
        return false;
    }
    device.wall_soa.position = pos_data;

    return true;
}


static bool allocate_entity_vec_2d_r32(AppState& state, DeviceMemory& device)
{
    auto& buffer = state.device_entity_vec_2d_r32_buffer;

    if(!cuda::device_malloc(buffer, SIZE::Entity_Vec2Dr32))
    {
        print_error("entity Vec2Dr32");
        return false;
    }

    auto vec_data = cuda::push_elements(buffer, COUNT::PLAYER_ENTITIES);
    if(!vec_data)
    {
        print_error("player dim_m data");
        return false;
    }
    device.player_soa.dim_m = vec_data;

    vec_data = cuda::push_elements(buffer, COUNT::PLAYER_ENTITIES);
    if(!vec_data)
    {
        print_error("player dt data");
        return false;
    }
    device.player_soa.dt = vec_data;

    vec_data = cuda::push_elements(buffer, COUNT::PLAYER_ENTITIES);
    if(!vec_data)
    {
        print_error("player delta_pos_m data");
        return false;
    }
    device.player_soa.delta_pos_m = vec_data;

    vec_data = cuda::push_elements(buffer, COUNT::BLUE_ENTITIES);
    if(!vec_data)
    {
        print_error("blue dim_m data");
        return false;
    }
    device.blue_soa.dim_m = vec_data;

    vec_data = cuda::push_elements(buffer, COUNT::BLUE_ENTITIES);
    if(!vec_data)
    {
        print_error("blue dt data");
        return false;
    }
    device.blue_soa.dt = vec_data;

    vec_data = cuda::push_elements(buffer, COUNT::BLUE_ENTITIES);
    if(!vec_data)
    {
        print_error("blue delta_pos_m data");
        return false;
    }
    device.blue_soa.delta_pos_m = vec_data;

    vec_data = cuda::push_elements(buffer, COUNT::WALL_ENTITIES);
    if(!vec_data)
    {
        print_error("wall dim_m data");
        return false;
    }
    device.wall_soa.dim_m = vec_data;

    return true;
}


bool init_device_memory(AppState& state, app::ScreenBuffer& buffer)
{
    assert(sizeof(Pixel) == buffer.bytes_per_pixel);

    auto const width = SCREEN_WIDTH_PX;
    auto const height = SCREEN_HEIGHT_PX;

    assert(width == buffer.width);
    assert(height == buffer.height);

    DeviceMemory device{};

    // tiles/pixels
    if(!cuda::device_malloc(state.device_pixel_buffer, SIZE::DeviceMemory_Pixel))
    {
        print_error("device pixel_buffer");
        return false;
    }

    if(!load_device_assets(state.device_pixel_buffer, device.assets))
    {
        print_error("device assets");
        return false;
    }

    auto& screen = device.screen_pixels;
    if(!init_image(screen, state.device_pixel_buffer, width, height))
    {
        print_error("screen_pixels");
        return false;
    }

    state.screen_pixels.data = (Pixel*)buffer.memory;
    state.screen_pixels.width = width;
    state.screen_pixels.height = height;

    state.device_pixels = device.screen_pixels;

    // tilemap
    auto const n_tilemap_tiles = WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE;

    if(!cuda::device_malloc(state.device_tile_buffer, SIZE::DeviceMemory_Tile))
    {
        print_error("device tiles");
        return false;
    }

    auto tilemap_data = cuda::push_elements(state.device_tile_buffer, n_tilemap_tiles);
    if(!tilemap_data)
    {
        print_error("tilemap data");
        return false;
    }
    device.tilemap.data = tilemap_data;
    device.tilemap.width = WORLD_WIDTH_TILE;
    device.tilemap.height = WORLD_HEIGHT_TILE;

/*
    // entities
    if(!cuda::device_malloc(state.device_entity_buffer, SIZE::DeviceMemory_Entity))
    {
        print_error("entities");
        return false;
    }

    auto player_data = cuda::push_elements(state.device_entity_buffer, COUNT::PLAYER_ENTITIES);
    if(!player_data)
    {
        print_error("player data");
        return false;
    }
    device.player_entities.data = player_data;
    device.player_entities.n_elements = COUNT::PLAYER_ENTITIES;

    auto blue_data = cuda::push_elements(state.device_entity_buffer, COUNT::BLUE_ENTITIES);
    if(!blue_data)
    {
        print_error("blue data");
        return false;
    }
    device.blue_entities.data = blue_data;
    device.blue_entities.n_elements = COUNT::BLUE_ENTITIES;

    auto wall_data = cuda::push_elements(state.device_entity_buffer, COUNT::WALL_ENTITIES);
    if(!wall_data)
    {
        print_error("wall data");
        return false;
    }
    device.wall_entities.data = wall_data;
    device.wall_entities.n_elements = COUNT::WALL_ENTITIES;

    device.entities.data = player_data;
*/

    // entity soa
    if(!allocate_entity_status(state, device))
    {
        return false;
    }
    
    if(!allocate_entity_image(state, device))
    {
        return false;
    }

    if(!allocate_entity_pixel(state, device))
    {
        return false;
    }

    if(!allocate_entity_r32(state, device))
    {
        return false;
    }    

    if(!allocate_entity_world_position(state, device))
    {
        return false;
    }    
   
    if(!allocate_entity_vec_2d_r32(state, device))
    {
        return false;
    }    
    
    if(!cuda::device_malloc(state.device_buffer, 1))
    {
        print_error("state.device_buffer");
        return false;
    }

    if(!cuda::memcpy_to_device(&device, state.device_buffer.data, sizeof(DeviceMemory)))
    {
        print_error("memcpy device");
		return false;
    }

    return true;
}


bool init_unified_memory(AppState& state)
{
    UnifiedMemory unified{};

    unified.frame_count = 0;
    unified.user_player_id = 0;   

    auto const n_input_records = 2 * INPUT::MAX_RECORDS;
    if(!cuda::unified_malloc(state.unified_input_record_buffer, n_input_records))
    {
        print_error("unified_input_records");
        return false;
    }

    if(!init_input_list(unified.current_inputs, state.unified_input_record_buffer))
    {
        print_error("current_inputs");
        return false;
    }

    if(!init_input_list(unified.previous_inputs, state.unified_input_record_buffer))
    {
        print_error("previous_inputs");
        return false;
    }

    if(!cuda::unified_malloc(state.unified_buffer, 1))
    {
        print_error("state.unified_buffer");
        return false;
    }    

    *state.unified_buffer.data = unified;

    return true;
}


void init_app_input(AppInput& app_input)
{
    //app_input.frame_count = 0;
    app_input.reset_frame_count = false;
    
    app_input.screen_width_m = MIN_SCREEN_WIDTH_M;
    app_input.screen_height_m = screen_height_m(app_input.screen_width_m);

    app_input.screen_position.tile = { 0, 0 };
    app_input.screen_position.offset_m = { 0.0f, 0.0f };
}

