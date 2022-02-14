#include "app.hpp"
#include "gpu/gpu_app.hpp"
#include "../libimage/libimage.hpp"

namespace img = libimage;

#include <cassert>
#include <cmath>

#define PRINT

#ifdef PRINT
#include <cstdio>
#endif

static void print(const char* msg)
{
#ifdef PRINT

    printf("* %s *\n", msg);

#endif
}

constexpr auto GRASS_TILE_PATH = "/home/adam/Repos/GPUGame/assets/tiles/basic_grass.png";


constexpr r32 screen_height_m(r32 screen_width_m)
{
    return screen_width_m * app::SCREEN_BUFFER_HEIGHT / app::SCREEN_BUFFER_WIDTH;
}


static void init_state_props(StateProps& props)
{
    props.screen_width_px = app::SCREEN_BUFFER_WIDTH;
    props.screen_height_px = app::SCREEN_BUFFER_HEIGHT;

    props.screen_width_m = MIN_SCREEN_WIDTH_M;    

    props.screen_position.tile = { 10, 10 };
    props.screen_position.offset_m = { 0.0f, 0.0f };
    
    props.player_direction = { 0.0f, 0.0f };
}


static bool load_tile_assets(DeviceMemory& device)
{    
    Image read_img;
    Image tile_img{};
    tile_img.width = TILE_WIDTH_PX;
    tile_img.height = TILE_HEIGHT_PX;

    auto& tiles = device.tiles;

    if(!make_device_tile(tiles.grass, device.buffer))
    {
        print("make grass tile failed");
        return false;
    }

    if(!make_device_tile(tiles.white, device.buffer))
    {
        print("make white tile failed");
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

    // temp make white
    auto white = to_pixel(255, 255, 255);
    for(u32 i = 0; i < tile_img.width * tile_img.height; ++i)
    {
        tile_img.data[i] = white;
    }

    if(!copy_to_device(tile_img, tiles.white))
    {
        print("copy white tile failed");
        cleanup();
        return false;
    }

    cleanup();
    return true;
}


static bool init_device_memory(DeviceMemory& device, u32 screen_width, u32 screen_height)
{
    u32 n_world_tiles = WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE;  
    auto world_tile_sz = n_world_tiles * sizeof(DeviceTile);

    auto entity_sz = N_ENTITIES * sizeof(Entity);

    auto tile_asset_sz = N_TILE_BITMAPS * (TILE_HEIGHT_PX * TILE_WIDTH_PX * sizeof(Pixel) + sizeof(Pixel));

    auto required_sz = world_tile_sz + entity_sz + tile_asset_sz;

    if(!device_malloc(device.buffer, required_sz))
    {
        return false;
    }

    if(!make_device_array(device.entities, N_ENTITIES, device.buffer))
    {
        return false;
    }

    if(!make_device_matrix(device.tilemap, WORLD_WIDTH_TILE, WORLD_HEIGHT_TILE, device.buffer))
    {
        return false;
    }

    if(!load_tile_assets(device))
    {
        return false;
    }

    gpu::init_device_memory(device);

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


static WorldPosition add_delta(WorldPosition const& pos, Vec2Dr32 const& delta)
{
    WorldPosition added = pos;

    apply_delta(added, delta);

    return added;
}


static void update_screen_position(WorldPosition& screen_pos, Vec2Dr32 const& delta, r32 screen_width_m)
{
    apply_delta(screen_pos, delta);
}


static void process_input(Input const& input, AppState& state)
{
    auto& controller = input.controllers[0];
    auto& keyboard = input.keyboard;
    auto& props = state.props;
    auto& player_d = state.props.player_direction;

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
    if(props.screen_width_m < MAX_SCREEN_WIDTH_M && controller.stick_right_y.end < -0.5f)
    {
        auto old_w = props.screen_width_m;
        auto old_h = screen_height_m(old_w);

        props.screen_width_m = std::min(props.screen_width_m + zoom_m, MAX_SCREEN_WIDTH_M);

        auto new_w = props.screen_width_m;
        auto new_h = screen_height_m(new_w);

        camera_d_m.x += 0.5f * (old_w - new_w);
        camera_d_m.y += 0.5f * (old_h - new_h);
    }

    auto dist_m = dt * 1.5f;
    player_d = { 0.0f, 0.0f };

    if(controller.dpad_up.is_down)
    {
        player_d.y -= dist_m;
    }
    if(controller.dpad_down.is_down)
    {
        player_d.y += dist_m;
    }
    if(controller.dpad_left.is_down)
    {
        player_d.x -= dist_m;
    }
    if(controller.dpad_right.is_down)
    {
        player_d.x += dist_m;
    }

    if(player_d.x != 0.0f && player_d.y != 0.0f)
    {
        player_d.x *= 0.707107;
        player_d.y *= 0.707107;
    }

    if(controller.button_b.pressed)
    {
        platform_signal_stop();
    }

    update_screen_position(props.screen_position, camera_d_m, props.screen_width_m);
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

        gpu::update(state);
        gpu::render(state);
    }

	
	void end_program(AppMemory& memory)
    {
        auto& state = get_state(memory);

        device_free(state.device.buffer);
		device_free(state.unified.buffer);
    }
}