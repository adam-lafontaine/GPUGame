#include "app_include.hpp"
#include "../libimage/libimage.hpp"

namespace img = libimage;


static Pixel get_avg_color(image_t const& image)
{
    auto sub_h = image.height / 10;
    auto sub_w = image.width / 10;

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


static bool init_tile(MemoryBuffer<Pixel>& buffer, Image const& host_image, Tile& tile)
{
    auto const n_pixels = host_image.width * host_image.height;

    tile.bitmap_data = cuda::push_elements(buffer, n_pixels);
    if(!tile.bitmap_data)
    {
        print_error("tile.bitmap_data");
        return false;
    }

    if(!cuda::memcpy_to_device(host_image.data, tile.bitmap_data, n_pixels * sizeof(Pixel)))
    {
        print_error("tile memcpy");
        return false;
    }

    auto avg_image_color = get_avg_color(host_image);    

    tile.avg_color = cuda::push_elements(buffer, 1);
    if(!tile.avg_color)
    {
        print_error("tile.avg_color");
        return false;
    }

    if(!cuda::memcpy_to_device(&avg_image_color, tile.avg_color, sizeof(Pixel)))
    {
        print_error("avg_color memcpy");
        return false;
    }

    return true;
}


static bool load_device_assets(MemoryBuffer<Pixel>& buffer, DeviceAssets& assets)
{
    Image read_img{};
    Image tile_img{};
    tile_img.width = TILE_WIDTH_PX;
    tile_img.height = TILE_HEIGHT_PX;

    auto const cleanup = [&]()
    {
        img::destroy_image(read_img);
        img::destroy_image(tile_img);
    };

    img::read_image_from_file(GRASS_TILE_PATH, read_img);
    img::resize_image(read_img, tile_img);

    if(!init_tile(buffer, tile_img, assets.grass_tile))
    {
        cleanup();
        print_error("init grass");
        return false;
    }

    // temp make brown
    auto brown = to_pixel(150, 75, 0);
    for(u32 i = 0; i < tile_img.width * tile_img.height; ++i)
    {
        tile_img.data[i] = brown;
    }

    if(!init_tile(buffer, tile_img, assets.brown_tile))
    {
        cleanup();
        print_error("init brown");
        return false;
    }


    // temp make black
    auto black = to_pixel(0, 0, 0);
    for(u32 i = 0; i < tile_img.width * tile_img.height; ++i)
    {
        tile_img.data[i] = black;
    }

    if(!init_tile(buffer, tile_img, assets.black_tile))
    {
        cleanup();
        print_error("init black");
        return false;
    }

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


bool init_device_memory(AppState& state)
{
    DeviceMemory device{};

    auto const n_pixels_per_tile = N_TILE_PIXELS;     
    auto const n_asset_tiles = N_TILE_BITMAPS;
    

    // tiles/pixels
    auto const  n_pixels = n_pixels_per_tile * n_asset_tiles;

    if(!cuda::device_malloc(state.device_pixel_buffer, n_pixels * sizeof(Pixel)))
    {
        print_error("device pixel_buffer");
        return false;
    }

    if(!load_device_assets(state.device_pixel_buffer, device.assets))
    {
        print_error("device assets");
        return false;
    }

    // tilemap
    auto const n_tilemap_tiles = WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE;

    if(!cuda::device_malloc(state.device_tile_buffer, n_tilemap_tiles * sizeof(Tile)))
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


    // entities
    auto const n_entities = N_BLUE_ENTITIES + N_BROWN_ENTITIES;

    if(!cuda::device_malloc(state.device_entity_buffer, n_entities * sizeof(Entity)))
    {
        print_error("entities");
        return false;
    }

    auto blue_data = cuda::push_elements(state.device_entity_buffer, N_BLUE_ENTITIES);
    if(!blue_data)
    {
        print_error("blue data");
        return false;
    }
    device.blue_entities.data = blue_data;
    device.blue_entities.n_elements = N_BLUE_ENTITIES;

    auto wall_data = cuda::push_elements(state.device_entity_buffer, N_BROWN_ENTITIES);
    if(!wall_data)
    {
        print_error("wall data");
        return false;
    }
    device.wall_entities.data = wall_data;
    device.wall_entities.n_elements = N_BROWN_ENTITIES;


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


bool init_unified_memory(AppState& state, app::ScreenBuffer& buffer)
{
    assert(sizeof(Pixel) == buffer.bytes_per_pixel);

    UnifiedMemory unified{};

    unified.frame_count = 0;    
    
    auto const width = buffer.width;
    auto const height = buffer.height;

    auto const n_pixels = width * height;
    if(!cuda::unified_malloc(state.unified_pixel_buffer, n_pixels))
    {
        print_error("unified_pixel");
        return false;
    }

    auto& screen = unified.screen_pixels;
    if(!init_image(screen, state.unified_pixel_buffer, width, height))
    {
        print_error("screen_pixels");
        return false;
    }    

    buffer.memory = screen.data;

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

    app_input.screen_width_px = app::SCREEN_BUFFER_WIDTH;
    app_input.screen_height_px = app::SCREEN_BUFFER_HEIGHT;

    app_input.screen_width_m = MIN_SCREEN_WIDTH_M;    

    app_input.screen_position.tile = { 0, 0 };
    app_input.screen_position.offset_m = { 0.0f, 0.0f };
}

