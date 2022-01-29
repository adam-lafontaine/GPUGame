// D-Bus not build with -rdynamic...
// sudo killall ibus-daemon

#include "../app/app.hpp"
#include "../utils/stopwatch.hpp"
#include "sdl_input.hpp"

#include <cstdio>
#include <thread>

constexpr auto WINDOW_TITLE = app::APP_TITLE;
constexpr int WINDOW_WIDTH = app::BUFFER_WIDTH;
constexpr int WINDOW_HEIGHT = app::BUFFER_HEIGHT;

constexpr u32 SCREEN_BYTES_PER_PIXEL = 4;

constexpr size_t APP_MEMORY_SIZE = Megabytes(16);

// control the framerate of the application
constexpr r32 TARGET_FRAMERATE_HZ = 60.0f;
constexpr r32 TARGET_MS_PER_FRAME = 1000.0f / TARGET_FRAMERATE_HZ;

GlobalVariable b32 g_running = false;


class BitmapBuffer
{
public:
    u32 bytes_per_pixel;

    void* memory;
    int width;
    int height;

    SDL_Renderer* renderer;
    SDL_Texture* texture;
};


static void allocate_app_memory(app::AppMemory& memory)
{
    memory.permanent_storage_size = APP_MEMORY_SIZE;

    size_t total_size = memory.permanent_storage_size;

    memory.permanent_storage = malloc(total_size);
}


static void destroy_app_memory(app::AppMemory& memory)
{
    if(memory.permanent_storage)
    {
        free(memory.permanent_storage);
    }    
}


static void open_game_controllers(SDLInput& sdl, Input& input)
{
    int num_joysticks = SDL_NumJoysticks();
    int c = 0;
    for(int j = 0; j < num_joysticks; ++j)
    {
        if (!SDL_IsGameController(j))
        {
            continue;
        }

        printf("found a controller\n");

        sdl.controllers[c] = SDL_GameControllerOpen(j);
        auto joystick = SDL_GameControllerGetJoystick(sdl.controllers[c]);
        if(!joystick)
        {
            printf("no joystick\n");
        }

        sdl.rumbles[c] = SDL_HapticOpenFromJoystick(joystick);
        if(!sdl.rumbles[c])
        {
            printf("no rumble from joystick\n");
        }
        else if(SDL_HapticRumbleInit(sdl.rumbles[c]))
        {
            printf("%s\n", SDL_GetError());
            SDL_HapticClose(sdl.rumbles[c]);
            sdl.rumbles[c] = 0;
        }
        else
        {
            printf("found a rumble\n");
        }

        ++c;

        if (c >= MAX_CONTROLLERS)
        {
            break;
        }
    }

    input.num_controllers = c;
}


static void close_game_controllers(SDLInput& sdl, Input const& input)
{
    for(int c = 0; c < input.num_controllers; ++c)
    {
        if(sdl.rumbles[c])
        {
            SDL_HapticClose(sdl.rumbles[c]);
        }
        SDL_GameControllerClose(sdl.controllers[c]);
    }
}


static void init_app_screen_buffer(app::ScreenBuffer& app_screen_buffer)
{
    app_screen_buffer.width = WINDOW_WIDTH;
    app_screen_buffer.height = WINDOW_HEIGHT;
    app_screen_buffer.bytes_per_pixel = SCREEN_BYTES_PER_PIXEL;
}


static bool init_bitmap_buffer(BitmapBuffer& buffer, app::ScreenBuffer const& screen_buffer, SDL_Window* window)
{
    buffer.width = (int)screen_buffer.width;
    buffer.height = (int)screen_buffer.height;
    buffer.bytes_per_pixel = screen_buffer.bytes_per_pixel;

    buffer.memory = screen_buffer.memory;

    buffer.renderer = SDL_CreateRenderer(window, -1, 0);
    
    if(!buffer.renderer)
    {
        printf("SDL_CreateRenderer failed\n%s\n", SDL_GetError());
        return false;
    }    

    if(buffer.texture)
    {
        SDL_DestroyTexture(buffer.texture);
    }

    buffer.texture = SDL_CreateTexture(
        buffer.renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        buffer.width,
        buffer.height);

    if(!buffer.texture)
    {
        printf("SDL_CreateTexture failed\n%s\n", SDL_GetError());
        return false;
    }    

    return true;
}


static void destroy_bitmap_buffer(BitmapBuffer& buffer)
{
    if(buffer.texture)
    {
        SDL_DestroyTexture(buffer.texture);
    }
}


u32 platform_to_color_32(u8 red, u8 green, u8 blue)
{
    return red << 16 | green << 8 | blue;
}


void platform_signal_stop()
{
    g_running = false;
}


static void end_program(app::AppMemory& memory)
{
    g_running = false;
    app::end_program(memory);
}


static void display_error(const char* msg)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "ERROR", msg, 0);
}


static void display_bitmap_in_window(BitmapBuffer const& buffer)
{
    auto error = SDL_UpdateTexture(buffer.texture, 0, buffer.memory, buffer.width * buffer.bytes_per_pixel);
    if(error)
    {
        printf("%s\n", SDL_GetError());
    }

    SDL_RenderCopy(buffer.renderer, buffer.texture, 0, 0);
    
    SDL_RenderPresent(buffer.renderer);
}


static void handle_sdl_window_event(SDL_WindowEvent const& w_event)
{
    auto window = SDL_GetWindowFromID(w_event.windowID);
    //auto renderer = SDL_GetRenderer(window);

    switch(w_event.event)
    {
        case SDL_WINDOWEVENT_SIZE_CHANGED:
        {

        }break;
        case SDL_WINDOWEVENT_EXPOSED:
        {
            
        } break;
    }
}


static void handle_sdl_event(SDL_Event const& event)
{
    switch(event.type)
    {
        case SDL_WINDOWEVENT:
        {
            handle_sdl_window_event(event.window);
        }break;
        case SDL_QUIT:
        {
            printf("SDL_QUIT\n");
            g_running = false;
        } break;
        case SDL_KEYDOWN:
        case SDL_KEYUP:
        {
            auto key_code = event.key.keysym.sym;
            auto alt = event.key.keysym.mod & KMOD_ALT;
            if(key_code == SDLK_F4 && alt)
            {
                printf("ALT F4\n");
                g_running = false;
            }
            else if(key_code == SDLK_ESCAPE)
            {
                printf("ESC\n");
                g_running = false;
            }

        } break;
        
    }
}


static bool init_sdl()
{
    auto sdl_options = 
        SDL_INIT_VIDEO | 
        SDL_INIT_GAMECONTROLLER | 
        SDL_INIT_HAPTIC;    
    
    if (SDL_Init(sdl_options) != 0)
    {
        printf("SDL_Init failed\n%s\n", SDL_GetError());
        return false;
    }

    return true;
}


static void close_sdl()
{
    SDL_Quit();
}


int main(int argc, char *argv[])
{
    printf("\n");
    if(!init_sdl())
    {
        display_error("Init SDL failed");
        return EXIT_FAILURE;
    }

    auto window = SDL_CreateWindow(
                    WINDOW_TITLE,
                    SDL_WINDOWPOS_UNDEFINED,
                    SDL_WINDOWPOS_UNDEFINED,
                    WINDOW_WIDTH,
                    WINDOW_HEIGHT,
                    SDL_WINDOW_RESIZABLE);
    if(!window)
    {
        display_error("SDL_CreateWindow failed");
        return EXIT_FAILURE;
    }

    app::AppMemory app_memory = {};
    app::ScreenBuffer app_screen_buffer = {};
    BitmapBuffer back_buffer = {};
    Input input[2] = {};
    SDLInput sdl_input = {};

    auto const cleanup = [&]()
    {
        close_game_controllers(sdl_input, input[0]);
        close_sdl();
        destroy_bitmap_buffer(back_buffer);
        destroy_app_memory(app_memory);
    };

    open_game_controllers(sdl_input, input[0]);
    input[1].num_controllers = input[0].num_controllers;
    printf("controllers = %d\n", input[0].num_controllers);    
    
    allocate_app_memory(app_memory);
    if (!app_memory.permanent_storage)
    {
        display_error("Allocating application memory failed");
        cleanup();
        return EXIT_FAILURE;
    }

    init_app_screen_buffer(app_screen_buffer);

    if(!app::initialize_memory(app_memory, app_screen_buffer))
    {
        display_error("Initializing application memory failed");
        cleanup();
        return EXIT_FAILURE;
    }
    
    if(!init_bitmap_buffer(back_buffer, app_screen_buffer, window))
    {
        display_error("Creating back buffer failed");
        cleanup();

        return EXIT_FAILURE;
    }    

    g_running = true;   
    
    bool in_current = 0;
    bool in_old = 1;
    Stopwatch sw;
    r64 frame_ms_elapsed = TARGET_MS_PER_FRAME;

    auto const wait_for_framerate = [&]()
    {
        frame_ms_elapsed = sw.get_time_milli();
        auto sleep_ms = (u32)(TARGET_MS_PER_FRAME - frame_ms_elapsed);
        if (frame_ms_elapsed < TARGET_MS_PER_FRAME && sleep_ms > 0)
        {    
            SDL_SetWindowTitle(window, WINDOW_TITLE);

            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            while (frame_ms_elapsed < TARGET_MS_PER_FRAME)
            {
                frame_ms_elapsed = sw.get_time_milli();
            }        
        }
        else
        {
            char buffer[30];
            snprintf(buffer, 30, "%s %d", WINDOW_TITLE, (int)frame_ms_elapsed);
            SDL_SetWindowTitle(window, buffer);
        }

        sw.start();
    };
    
    sw.start();
    while(g_running)
    {
        SDL_Event event;
        b32 has_event = SDL_PollEvent(&event);
        if(has_event)
        {            
            handle_sdl_event(event);
        }

        // does not miss frames but slows animation
        input[in_current].dt_frame = TARGET_MS_PER_FRAME / 1000.0f;

        // animation speed maintained but frames missed
        //input[in_current].dt_frame = frame_ms_elapsed / 1000.0f; // TODO:

        process_keyboard_input(has_event, event, input[in_old], input[in_current]);

        process_controller_input(sdl_input, input[in_old], input[in_current]);

        process_mouse_input(has_event, event, input[in_old], input[in_current]);

        app::update_and_render(app_memory, input[in_current]);

        wait_for_framerate();
        display_bitmap_in_window(back_buffer);

        // swap inputs
        in_current = in_old;
        in_old = !in_old;
    }
    
    app::end_program(app_memory);
    cleanup();

    return EXIT_SUCCESS;
}