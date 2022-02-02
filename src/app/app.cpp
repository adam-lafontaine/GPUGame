#include "app.hpp"
#include "render.hpp"

#include <cassert>



static void init_state_props(StateProps& props)
{
    props.screen_width_px = app::SCREEN_BUFFER_WIDTH;
    props.screen_width_m = tile_distance_m(MIN_SCREEN_WIDTH_TILE);
    

    props.screen_positon.tile = { 0, 0 };
    props.screen_positon.offset_m = { 0.0f, 0.0f };
}


static bool init_device_memory(DeviceMemory& device, u32 screen_width, u32 screen_height)
{
    u32 n_tiles = WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE;;
    auto tile_sz = n_tiles * sizeof(u32);

    auto required_sz = tile_sz;

    if(!device_malloc(device.buffer, required_sz))
    {
        return false;
    }

    if(!make_device_matrix(device.tilemap, WORLD_WIDTH_TILE, WORLD_HEIGHT_TILE, device.buffer))
    {
        return false;
    }

    init_device_memory(device);

    return true;
}


static bool init_unified_memory(UnifiedMemory& unified, u32 screen_width, u32 screen_height)
{
    auto const n_pixels = screen_width * screen_height;

    auto const screen_sz = sizeof(pixel_t) * n_pixels;

    auto unified_sz = screen_sz;
    if(!unified_malloc(unified.buffer, unified_sz))
    {
        return false;
    }

    if(!make_device_image(unified.screen_pixels, screen_width, screen_height, unified.buffer))
    {
        return false;
    }

    return true;
}


constexpr r32 px_to_m(r32 n_pixels, r32 width_m, u32 width_px)
{
    return n_pixels * width_m / width_px;
}


static void update_screen_position(WorldPosition& screen_pos, Vec2Dr32 const& delta, r32 screen_width_m)
{
    auto abs_pos_x = screen_pos.tile.x * TILE_LENGTH_M + screen_pos.offset_m.x + delta.x;
    auto abs_pos_y = screen_pos.tile.y * TILE_LENGTH_M + screen_pos.offset_m.y + delta.y;

    auto max_pos_x = WORLD_WIDTH_TILE * TILE_LENGTH_M - screen_width_m;
    auto screen_height_m = screen_width_m * app::SCREEN_BUFFER_HEIGHT / app::SCREEN_BUFFER_WIDTH;
    auto max_pos_y = WORLD_HEIGHT_TILE * TILE_LENGTH_M - screen_height_m;

    if(abs_pos_x < 0.0f)
    {
        abs_pos_x = 0.0f;
    }
    else if (abs_pos_x > max_pos_x)
    {
        abs_pos_x = max_pos_x;
    }    

    if(abs_pos_y < 0.0f)
    {
        abs_pos_y = 0.0f;
    }
    else if(abs_pos_y > max_pos_y)
    {
        abs_pos_y = max_pos_y;
    }

    screen_pos.tile.x = (u32)(abs_pos_x / TILE_LENGTH_M);
    screen_pos.offset_m.x = abs_pos_x - screen_pos.tile.x * TILE_LENGTH_M;

    screen_pos.tile.y = (u32)(abs_pos_y / TILE_LENGTH_M);
    screen_pos.offset_m.y = abs_pos_y - screen_pos.tile.y * TILE_LENGTH_M;
}


static void process_input(Input const& input, AppState& state)
{
    auto& controller = input.controllers[0];
    auto& keyboard = input.keyboard;
    auto& props = state.props;

    
    r32 camera_speed_px = 500.0f;

    auto dt = input.dt_frame;
    auto camera_movement_px = camera_speed_px * dt;
    auto camera_movement_m = px_to_m(camera_movement_px, props.screen_width_m, props.screen_width_px);

    Vec2Dr32 camera_d_m = { 0.0f, 0.0f };

    if(keyboard.up_key.is_down)
    {
        camera_d_m.y -= camera_movement_m;
    }

    if(keyboard.down_key.is_down)
    {
        camera_d_m.y += camera_movement_m;
    }

    if(keyboard.left_key.is_down)
    {
        camera_d_m.x -= camera_movement_m;
    }

    if(keyboard.right_key.is_down)
    {
        camera_d_m.x += camera_movement_m;
    }

    update_screen_position(props.screen_positon, camera_d_m, props.screen_width_m);
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
        
        u32 n_elements = 128; // just because
        auto elements_sz = n_elements * sizeof(u32);

        auto required_sz = state_sz + elements_sz;

        assert(required_sz <= memory.permanent_storage_size);

        auto& state = get_state(memory);

        auto mem = (u8*)memory.permanent_storage + state_sz;

        state.host.elements = (u32*)mem;
        state.host.n_elements = n_elements;

        mem += elements_sz;

        init_state_props(state.props);

        return state;
    }

    
	bool initialize_memory(AppMemory& memory, ScreenBuffer& screen)
    {
        assert(sizeof(pixel_t) == screen.bytes_per_pixel);

        memory.is_app_initialized = false;

        auto& state = get_initial_state(memory);        

        if(!init_unified_memory(state.unified, screen.width, screen.height))
		{
			return false;
		}        

        screen.memory = state.unified.screen_pixels.data;

        if(!init_device_memory(state.device, screen.width, screen.height))
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
        process_input(input, state);

        render(state);
    }

	
	void end_program(AppMemory& memory)
    {
        auto& state = get_state(memory);

        device_free(state.device.buffer);
		device_free(state.unified.buffer);
    }
}