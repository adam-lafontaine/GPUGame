#include "app.hpp"
#include "app_types.hpp"

#include <cassert>


static constexpr size_t app_state_required_size()
{
    return sizeof(AppState);
}


static constexpr size_t device_memory_size()
{
    return 10; // TODO
}


static void process_input(Input const& input, AppState& state)
{

}


void render(AppState& state)
{

}


namespace app
{

    static AppState& get_state(AppMemory const& memory)
    {
        auto& state = *(AppState*)memory.permanent_storage;

		return state;
    }


    static AppState& get_state(AppMemory& memory, ScreenBuffer const& buffer)
    {
        assert(app_state_required_size() <= memory.permanent_storage_size);

        return get_state(memory);
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
        if(!device_malloc(device.buffer, device_memory_size()))
        {
            return false;
        }

        return true;
    }



    
	bool initialize_memory(AppMemory& memory, ScreenBuffer& screen)
    {
        auto& state = get_state(memory, screen);

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