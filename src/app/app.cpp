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


static void process_input(Input const& input, AppState& state)
{
    auto& controller = input.controllers[0];
    auto& props = state.props;
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