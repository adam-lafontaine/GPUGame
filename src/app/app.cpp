#include "app.hpp"
#include "render.hpp"

#include <cassert>


static void process_input(Input const& input, AppState& state)
{
    auto& controller = input.controllers[0];
    auto& props = state.props;

    if(controller.button_b.pressed)
    {
        props.red += 10;
    }

    if(controller.button_a.pressed)
    {
        props.green += 10;
    }

    if(controller.button_x.pressed)
    {
        props.blue += 10;
    }

    if(controller.button_y.pressed)
    {
        props.red = 255;
        props.green = 255;
        props.blue = 255;
    }
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

        init_state_props(state);

        return state;
    }


    static bool init_unified_memory(UnifiedMemory& unified, ScreenBuffer& screen)
	{
		assert(sizeof(pixel_t) == screen.bytes_per_pixel);

		auto const width = screen.width;
		auto const height = screen.height;

        auto const n_pixels = width * height;

		auto const screen_sz = sizeof(pixel_t) * n_pixels;

		auto unified_sz = screen_sz;
		if(!unified_malloc(unified.buffer, unified_sz))
		{
			return false;
		}

        if(!make_device_image(unified.screen_pixels, width, height, unified.buffer))
        {
            return false;
        }

		screen.memory = unified.screen_pixels.data;

		return true;
	}


    static bool init_device_memory(DeviceMemory& device, ScreenBuffer const& buffer)
    {
        u32 n_elements = 128; // just because
        auto elements_sz = n_elements * sizeof(r32);

        auto required_sz = elements_sz;

        if(!device_malloc(device.buffer, required_sz))
        {
            return false;
        }

        if(!make_device_array(device.r32_array, n_elements, device.buffer))
        {
            return false;
        }

        return true;
    }

    
	bool initialize_memory(AppMemory& memory, ScreenBuffer& screen)
    {
        memory.is_app_initialized = false;

        auto& state = get_initial_state(memory);        

        if(!init_unified_memory(state.unified, screen))
		{
			return false;
		}

        if(!init_device_memory(state.device, screen))
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