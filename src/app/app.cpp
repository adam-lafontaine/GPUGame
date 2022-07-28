#include "app_include.hpp"




static void add_input_record(InputList& list, InputRecord& item)
{
    assert(list.data);
    assert(list.size < list.capacity);
    assert(item.frame_begin);
    assert(item.input);

    list.data[list.size++] = item;
}


static InputRecord& get_last_input_record(InputList const& list)
{
    assert(list.data);
    assert(list.size);

    return list.data[list.size - 1];
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
    auto camera_movement_m = px_to_m(camera_movement_px, app_input.screen_width_m, state.screen_pixels.width);    

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

        if(!init_unified_memory(state))
        {
            return false;
        }

        if(!init_device_memory(state, screen))
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

        auto src = state.device_pixels.data;
        auto dst = state.screen_pixels.data;
        auto size = state.device_pixels.width * state.device_pixels.height * sizeof(Pixel);

        cuda::memcpy_to_host(src, dst, size);
    }

	
	void end_program(AppMemory& memory)
    {
        auto& state = get_state(memory);  

        cuda::free(state.device_buffer);
        cuda::free(state.device_pixel_buffer);
        cuda::free(state.device_tile_buffer);
        cuda::free(state.device_entity_buffer);
        
        cuda::free(state.unified_input_record_buffer);
        cuda::free(state.unified_buffer);

    }
}