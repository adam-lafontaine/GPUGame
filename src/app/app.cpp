#include "app_include.hpp"

#ifndef NDEBUG
#define PRINT_APP_ERROR
#endif

#ifdef PRINT_APP_ERROR
#include <cstdio>
#endif

static void print_error(cstr msg)
{
#ifdef PRINT_APP_ERROR
	printf("\n*** APP ERROR ***\n\n");
	printf("%s", msg);
	printf("\n\n******************\n\n");
#endif
}


void add_input_record(DeviceInputList& list, InputRecord& item)
{
    assert(list.data);
    assert(list.size < list.capacity);
    assert(item.frame_begin);
    assert(item.input);

    list.data[list.size++] = item;
}


InputRecord& get_last_input_record(DeviceInputList const& list)
{
    assert(list.data);
    assert(list.size);

    return list.data[list.size - 1];
}


static void init_app_input(AppInput& app_input)
{
    //app_input.frame_count = 0;
    app_input.reset_frame_count = false;

    app_input.screen_width_px = app::SCREEN_BUFFER_WIDTH;
    app_input.screen_height_px = app::SCREEN_BUFFER_HEIGHT;

    app_input.screen_width_m = MIN_SCREEN_WIDTH_M;    

    app_input.screen_position.tile = { 0, 0 };
    app_input.screen_position.offset_m = { 0.0f, 0.0f };
}


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


static bool init_tile(MemoryBuffer<Pixel>& buffer, Image const& host_image, DeviceTile& tile)
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


static bool init_device_memory(AppState& state)
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

    if(!cuda::device_malloc(state.device_tile_buffer, n_tilemap_tiles * sizeof(DeviceTile)))
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


static bool init_input_list(DeviceInputList& list, MemoryBuffer<InputRecord>& buffer)
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


static bool init_unified_memory(AppState& state, app::ScreenBuffer& buffer)
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


static void apply_delta(WorldPosition& pos, Vec2Dr32 const& delta)
{
    if(delta.x != 0.0f)
    {
        r32 dist_m = pos.offset_m.x + delta.x;
        i32 delta_tile = floor_r32_to_i32(dist_m / TILE_LENGTH_M);
        
        pos.tile.x = pos.tile.x + delta_tile;
        pos.offset_m.x = dist_m - (r32)delta_tile * TILE_LENGTH_M;
    }    

    if(delta.y != 0.0f)
    {
        r32 dist_m = pos.offset_m.y + delta.y;
        i32 delta_tile = floor_r32_to_i32(dist_m / TILE_LENGTH_M);
        
        pos.tile.y = pos.tile.y + delta_tile;
        pos.offset_m.y = dist_m - (r32)delta_tile * TILE_LENGTH_M;
    }    
}


static void process_camera_input(Input const& input, AppState& state)
{
    auto& controller = input.controllers[0];
    auto& keyboard = input.keyboard;
    auto& app_input = state.app_input;

    auto dt = input.dt_frame;

    r32 max_camera_speed_px = 300.0f;
    r32 min_camera_speed_px = 200.0f;

    auto camera_speed_px = max_camera_speed_px - 
        (app_input.screen_width_m - MIN_SCREEN_WIDTH_M) / (MAX_SCREEN_WIDTH_M - MIN_SCREEN_WIDTH_M) * (max_camera_speed_px - min_camera_speed_px);
  
    auto camera_movement_px = camera_speed_px * dt;
    auto camera_movement_m = px_to_m(camera_movement_px, app_input.screen_width_m, app_input.screen_width_px);    

    Vec2Dr32 camera_d_m = { 0.0f, 0.0f };

    if(keyboard.up_key.is_down || controller.stick_left_y.end > 0.5f)
    {
        camera_d_m.y -= camera_movement_m;
    }
    if(keyboard.down_key.is_down || controller.stick_left_y.end < -0.5f)
    {
        camera_d_m.y += camera_movement_m;
    }
    if(keyboard.left_key.is_down || controller.stick_left_x.end < -0.5f)
    {
        camera_d_m.x -= camera_movement_m;
    }
    if(keyboard.right_key.is_down || controller.stick_left_x.end > 0.5f)
    {
        camera_d_m.x += camera_movement_m;
    }

    r32 zoom_speed = 50.0f;
    auto zoom_m = zoom_speed * dt;

    if(app_input.screen_width_m > MIN_SCREEN_WIDTH_M && controller.stick_right_y.end > 0.5f)
    {
        auto old_w = app_input.screen_width_m;
        auto old_h = screen_height_m(old_w);

        app_input.screen_width_m = std::max(app_input.screen_width_m - zoom_m, MIN_SCREEN_WIDTH_M);

        auto new_w = app_input.screen_width_m;
        auto new_h = screen_height_m(new_w);

        camera_d_m.x += 0.5f * (old_w - new_w);
        camera_d_m.y += 0.5f * (old_h - new_h);
    }
    else if(app_input.screen_width_m < MAX_SCREEN_WIDTH_M && controller.stick_right_y.end < -0.5f)
    {
        auto old_w = app_input.screen_width_m;
        auto old_h = screen_height_m(old_w);

        app_input.screen_width_m = std::min(app_input.screen_width_m + zoom_m, MAX_SCREEN_WIDTH_M);

        auto new_w = app_input.screen_width_m;
        auto new_h = screen_height_m(new_w);

        camera_d_m.x += 0.5f * (old_w - new_w);
        camera_d_m.y += 0.5f * (old_h - new_h);
    } 

    apply_delta(app_input.screen_position, camera_d_m);
}


static void process_player_input(Input const& input, AppState& state)
{
    auto& controller = input.controllers[0];
    auto& input_records = state.unified_buffer.data->current_inputs;
    //auto& keyboard = input.keyboard;
    auto& app_input = state.app_input;
    auto current_frame = (*state.unified_buffer.data).frame_count;

    uInput player_input = 0;

    if(controller.dpad_up.is_down)
    {
        player_input |= INPUT::PLAYER_UP;
    }
    if(controller.dpad_down.is_down)
    {
        player_input |= INPUT::PLAYER_DOWN;
    }
    if(controller.dpad_left.is_down)
    {
        player_input |= INPUT::PLAYER_LEFT;
    }
    if(controller.dpad_right.is_down)
    {
        player_input |= INPUT::PLAYER_RIGHT;
    }

    auto const create_record = [&]()
    {
        InputRecord r{};
        r.frame_begin = current_frame;
        r.frame_end = current_frame + 1;
        r.input = player_input;

        r.est_dt_frame = input.dt_frame; // avg?

        add_input_record(input_records, r);
    };

    if(!input_records.size)
    {
        if(player_input)
        {
            create_record();
        }

        return;
    }

    auto& last = get_last_input_record(input_records);

    // repeat input
    if(player_input && last.frame_end == current_frame && player_input == last.input)
    {
        ++last.frame_end;
    }

    // change input
    else if(player_input && last.frame_end == current_frame && player_input != last.input)
    {
        ++last.frame_end;
        create_record();
    }

    // end input
    else if(!player_input && last.frame_end == current_frame)
    {
        return;
    }

    // start input
    else if(player_input)
    {
        create_record();
    }
}


static void copy_inputs(AppState& state)
{
    auto& src = state.unified_buffer.data->current_inputs;
    auto& dst = state.unified_buffer.data->previous_inputs;

    assert(src.data);
    assert(dst.data);

    if(src.size == 0 && dst.size == 0)
    {
        return;
    }
    if(src.size == dst.size + 1)
    {
        dst.data[dst.size++] = src.data[src.size - 1];
    }
    else if(src.size == dst.size)
    {
        dst.data[dst.size - 1] = src.data[src.size - 1];
    }
    else
    {
        assert(false);
    }
}


static void process_input(Input const& input, AppState& state)
{
    process_camera_input(input, state);
    process_player_input(input, state);

    copy_inputs(state);

    auto& controller = input.controllers[0];
    auto& keyboard = input.keyboard;
    auto& app_input = state.app_input;

    if(controller.button_b.pressed)
    {
        platform_signal_stop();
    }

    if(controller.button_x.pressed)
    {
        
    }
    else
    {
        
    }
}


static void next_frame(AppState& state)
{
    auto& app_input = state.app_input;
    auto& unified = *state.unified_buffer.data;

    if(app_input.reset_frame_count)
    {
        unified.frame_count = 0;
        app_input.reset_frame_count = false;
    }

    ++unified.frame_count;
}


namespace app
{

    static AppState& get_state(AppMemory const& memory)
    {
        auto& state = *(AppState*)memory.permanent_storage;

		return state;
    }


    static AppState& get_initial_state(AppMemory& memory)
    {
        auto state_sz = sizeof(AppState);

        auto required_sz = state_sz + host_memory_sz();

        assert(required_sz <= memory.permanent_storage_size);

        auto& state = get_state(memory);
        init_app_input(state.app_input);

        return state;
    }

    
	bool initialize_memory(AppMemory& memory, ScreenBuffer& screen)
    {
        assert(sizeof(pixel_t) == screen.bytes_per_pixel);

        memory.is_app_initialized = false;

        auto& state = get_initial_state(memory);

        if(!init_unified_memory(state, screen))
        {
            return false;
        }

        if(!init_device_memory(state))
        {
            return false;
        }

        if(!gpu::init_device_memory(state))
        {
            return false;
        }

        memory.is_app_initialized = true;
        return true;
    }

	
	void update_and_render(AppMemory& memory, Input const& input)
    {
        if (!memory.is_app_initialized)
		{
			return;
		}

		auto& state = get_state(memory);
        next_frame(state);

        process_input(input, state); 

        gpu::update(state);
        gpu::render(state);        
    }

	
	void end_program(AppMemory& memory)
    {
        auto& state = get_state(memory);  

        cuda::free(state.device_buffer);
        cuda::free(state.device_pixel_buffer);
        cuda::free(state.device_tile_buffer);
        cuda::free(state.device_entity_buffer);

        cuda::free(state.unified_pixel_buffer);
        cuda::free(state.unified_input_record_buffer);
        cuda::free(state.unified_buffer);

    }
}