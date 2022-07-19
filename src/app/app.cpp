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


static bool load_device_assets(DeviceAssets& device_assets)
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

    if(!copy_to_device(tile_img, device_assets.grass_tile))
    {
        print("copy grass tile failed");
        cleanup();
        return false;
    }

    // temp make brown
    auto brown = to_pixel(150, 75, 0);
    for(u32 i = 0; i < tile_img.width * tile_img.height; ++i)
    {
        tile_img.data[i] = brown;
    }

    if(!copy_to_device(tile_img, device_assets.brown_tile))
    {
        print("copy brown tile failed");
        cleanup();
        return false;
    }

    // temp make black
    auto black = to_pixel(0, 0, 0);
    for(u32 i = 0; i < tile_img.width * tile_img.height; ++i)
    {
        tile_img.data[i] = black;
    }

    if(!copy_to_device(tile_img, device_assets.black_tile))
    {
        print("copy black tile failed");
        cleanup();
        return false;
    }

    cleanup();
    return true;
}


static bool init_device_memory(AppState& state)
{
    auto& buffer = state.device_buffer;

    if(!device::malloc(buffer, device_memory_total_size()))
    {
        return false;
    }

    DeviceMemoryOld device{};

    if(!make_device_memory(device, buffer))
    {
        return false;
    }

    if(!load_device_assets(device.assets))
    {
        return false;
    }

    auto struct_size = sizeof(DeviceMemoryOld);

    auto device_dst = device::push_bytes(buffer, struct_size);
    if(!device_dst)
    {
        return false;
    }

    //assert(buffer.size == buffer.capacity);

    if(!cuda::memcpy_to_device(&device, device_dst, struct_size))
    {
        return false;
    }

    state.device_p = (DeviceMemoryOld*)device_dst;

    return true;
}


static bool init_unified_memory_old(AppState& state)
{
    auto& buffer = state.unified_buffer;

    if(!device::unified_malloc(buffer, unified_memory_total_size(app::SCREEN_BUFFER_WIDTH, app::SCREEN_BUFFER_HEIGHT)))
    {
        return false;
    }

    UnifiedMemoryOld unified_p{};

    //unified_p.frame_count = 0;

    if(!make_unified_memory(unified_p, buffer, app::SCREEN_BUFFER_WIDTH, app::SCREEN_BUFFER_HEIGHT))
    {        
        return false;
    }

    auto struct_size = sizeof(UnifiedMemoryOld);

    auto device_dst = device::push_bytes(buffer, struct_size);

    if(!cuda::memcpy_to_device(&unified_p, device_dst, struct_size))
    {
        return false;
    }

    assert(buffer.size == buffer.capacity);

    state.unified_p = (UnifiedMemoryOld*)device_dst;


    UnifiedMemoryOld unified{};


    return true;
}


static bool init_unified_memory(AppState& state, app::ScreenBuffer& buffer)
{
    assert(sizeof(Pixel) == buffer.bytes_per_pixel);

    UnifiedMemory unified{};

    unified.frame_count = 0;

    
    auto& screen = unified.screen_pixels;

    auto const width = buffer.width;
    auto const height = buffer.height;

    auto const n_pixels = width * height;

    if(!cuda::unified_malloc(state.unified_pixel, n_pixels))
    {
        print_error("unified_pixel");
        return false;
    }

    assert(state.unified_pixel.data);

    screen.data = cuda::push_elements(state.unified_pixel, n_pixels);
    if(!screen.data)
    {
        print_error("screen data");
        return false;
    }

    screen.width = width;
    screen.height = height;

    buffer.memory = screen.data;
    

    if(!cuda::unified_malloc(state.unified, 1))
    {
        print_error("state.unified");
        return false;
    }    

    *state.unified.data = unified;    

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
    auto& input_records = state.unified_p->current_inputs;
    //auto& keyboard = input.keyboard;
    auto& app_input = state.app_input;
    auto current_frame = (*state.unified.data).frame_count;

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
    auto& src = state.unified_p->current_inputs;
    auto& dst = state.unified_p->previous_inputs;

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
    auto& unified = *state.unified.data;

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

        if(!init_unified_memory_old(state))
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

        //screen.memory = state.unified_p->screen_pixels.data;

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

        device::free(state.device_buffer);
		device::free(state.unified_buffer);
    }
}