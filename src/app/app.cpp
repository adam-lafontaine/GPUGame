#include "app_include.hpp"
/*
TileList* make_device_tile_list(device::MemoryBuffer& buffer)
{
    
}



DeviceInputList* make_device_input_list(device::MemoryBuffer& buffer)
{
    auto data_size = device_input_list_data_size();
    auto struct_size = sizeof(DeviceInputList);
    
    auto input_record_data = device::push_bytes(buffer, data_size);
    if(!input_record_data)
    {
        return nullptr;
    }

    auto struct_data = device::push_bytes(buffer, struct_size);
    if(!struct_data)
    {
        device::pop_bytes(buffer, data_size);
        return nullptr;
    }

    DeviceInputList list;
    list.capacity = MAX_INPUT_RECORDS;
    list.size = 0;
    list.read_index = 0;
    list.data = (InputRecord*)input_record_data;    

    auto device_dst = (DeviceInputList*)struct_data;

    if(!cuda_memcpy_to_device(&list, device_dst, struct_size))
    {
        return nullptr;
    }

    return device_dst;
}

*/


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


static void init_state_props(StateProps& props)
{
    props.frame_count = 0;
    props.reset_frame_count = false;

    props.screen_width_px = app::SCREEN_BUFFER_WIDTH;
    props.screen_height_px = app::SCREEN_BUFFER_HEIGHT;

    props.screen_width_m = MIN_SCREEN_WIDTH_M;    

    props.screen_position.tile = { 0, 0 };
    props.screen_position.offset_m = { 0.0f, 0.0f };
}


static bool load_tile_assets(device::MemoryBuffer& buffer, DeviceMemory& device)
{    
    Image read_img;
    Image tile_img{};
    tile_img.width = TILE_WIDTH_PX;
    tile_img.height = TILE_HEIGHT_PX;

    auto& tiles = device.tile_assets;

    if(!device::push_device_tile(buffer, tiles.grass))
    {
        print("make grass tile failed");
        return false;
    }
    
    if(!device::push_device_tile(buffer, tiles.brown))
    {
        print("make brown tile failed");
        return false;
    }

    if(!device::push_device_tile(buffer, tiles.black))
    {
        print("make black tile failed");
        return false;
    }

    auto const cleanup = [&]()
    {
        img::destroy_image(read_img);
        img::destroy_image(tile_img);
    };

    img::read_image_from_file(GRASS_TILE_PATH, read_img);
    img::resize_image(read_img, tile_img);

    if(!copy_to_device(tile_img, tiles.grass))
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

    if(!copy_to_device(tile_img, tiles.brown))
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

    if(!copy_to_device(tile_img, tiles.brown))
    {
        print("copy black tile failed");
        cleanup();
        return false;
    }

    cleanup();
    return true;
}


static bool init_device_memory(device::MemoryBuffer& buffer, DeviceMemory& device)
{
    if(!device::malloc(buffer, device_memory_sz()))
    {
        return false;
    }

    // Note: order matters

    if(!device::push_device_matrix(buffer, device.tilemap, WORLD_WIDTH_TILE, WORLD_HEIGHT_TILE))
    {
        return false;
    }

    if(!device::push_device_array(buffer, device.entities, N_ENTITIES))
    {
        return false;
    }

    if(!load_tile_assets(buffer, device))
    {
        return false;
    }

    if(!device::push_device_image(buffer, device.screen_pixels, app::SCREEN_BUFFER_WIDTH, app::SCREEN_BUFFER_HEIGHT))
    {
        return false;
    }

    auto device_previous_inputs = make_device_input_list(buffer);
    if(!device_previous_inputs)
    {
        return false;
    }

    device.previous_inputs = device_previous_inputs;

    gpu::init_device_memory(device);

    return true;
}


static bool init_unified_memory(device::MemoryBuffer& buffer, UnifiedMemory& unified)
{
    if(!device::unified_malloc(buffer, unified_memory_sz()))
    {
        return false;
    }

    auto unified_current_inputs = make_device_input_list(buffer);
    if(!unified_current_inputs)
    {
        return false;
    }

    unified.current_inputs = unified_current_inputs;

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


static void process_screen_input(Input const& input, AppState& state)
{
    auto& controller = input.controllers[0];
    auto& keyboard = input.keyboard;
    auto& props = state.props;

    auto dt = input.dt_frame;

    r32 max_camera_speed_px = 300.0f;
    r32 min_camera_speed_px = 200.0f;

    auto camera_speed_px = max_camera_speed_px - 
        (props.screen_width_m - MIN_SCREEN_WIDTH_M) / (MAX_SCREEN_WIDTH_M - MIN_SCREEN_WIDTH_M) * (max_camera_speed_px - min_camera_speed_px);
  
    auto camera_movement_px = camera_speed_px * dt;
    auto camera_movement_m = px_to_m(camera_movement_px, props.screen_width_m, props.screen_width_px);    

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

    if(props.screen_width_m > MIN_SCREEN_WIDTH_M && controller.stick_right_y.end > 0.5f)
    {
        auto old_w = props.screen_width_m;
        auto old_h = screen_height_m(old_w);

        props.screen_width_m = std::max(props.screen_width_m - zoom_m, MIN_SCREEN_WIDTH_M);

        auto new_w = props.screen_width_m;
        auto new_h = screen_height_m(new_w);

        camera_d_m.x += 0.5f * (old_w - new_w);
        camera_d_m.y += 0.5f * (old_h - new_h);
    }
    else if(props.screen_width_m < MAX_SCREEN_WIDTH_M && controller.stick_right_y.end < -0.5f)
    {
        auto old_w = props.screen_width_m;
        auto old_h = screen_height_m(old_w);

        props.screen_width_m = std::min(props.screen_width_m + zoom_m, MAX_SCREEN_WIDTH_M);

        auto new_w = props.screen_width_m;
        auto new_h = screen_height_m(new_w);

        camera_d_m.x += 0.5f * (old_w - new_w);
        camera_d_m.y += 0.5f * (old_h - new_h);
    } 

    apply_delta(props.screen_position, camera_d_m);
}


static void process_player_input(Input const& input, AppState& state)
{
    auto& controller = input.controllers[0];
    auto& input_records = *state.unified.current_inputs;
    //auto& keyboard = input.keyboard;
    auto& props = state.props;

    uInput player_input = 0;

    if(controller.dpad_up.is_down)
    {
        player_input |= INPUT_PLAYER_UP;
    }
    if(controller.dpad_down.is_down)
    {
        player_input |= INPUT_PLAYER_DOWN;
    }
    if(controller.dpad_left.is_down)
    {
        player_input |= INPUT_PLAYER_LEFT;
    }
    if(controller.dpad_right.is_down)
    {
        player_input |= INPUT_PLAYER_RIGHT;
    }

    auto const create_record = [&]()
    {
        InputRecord r{};
        r.frame_begin = props.frame_count;
        r.frame_end = props.frame_count + 1;
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
    auto current_frame = props.frame_count;

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


static void process_input(Input const& input, AppState& state)
{
    process_screen_input(input, state);
    process_player_input(input, state);

    auto& controller = input.controllers[0];
    auto& keyboard = input.keyboard;
    auto& props = state.props;

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
    auto& props = state.props;

    if(props.reset_frame_count)
    {
        props.frame_count = 0;
        props.reset_frame_count = false;
    }

    ++props.frame_count;
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
        init_state_props(state.props);

        auto mem = (u8*)memory.permanent_storage;
        auto offset = state_sz;

        auto& screen = state.host.screen_pixels;

        screen.width = state.props.screen_width_px; // needed?
        screen.height = state.props.screen_height_px;        
        screen.data = (Pixel*)(mem + offset);
        
        offset += screen.width * screen.height * sizeof(Pixel);

        return state;
    }

    
	bool initialize_memory(AppMemory& memory, ScreenBuffer& screen)
    {
        assert(sizeof(pixel_t) == screen.bytes_per_pixel);

        memory.is_app_initialized = false;

        auto& state = get_initial_state(memory);

        if(!init_unified_memory(state.unified_buffer, state.unified))
		{
			return false;
		}

        if(!init_device_memory(state.device_buffer, state.device))
        {
            return false;
        }      

        screen.memory = state.host.screen_pixels.data;    

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