#pragma once

#include "../../input/input.hpp"


#if defined(_WIN32)
#define SDL_MAIN_HANDLED
#endif

#include <SDL2/SDL.h>

#define PRINT_MESSAGES

#ifdef PRINT_MESSAGES
#include <cstdio>
#endif

//#define SDL2_IMPL_B

class SDLControllerInput
{
public:
    SDL_GameController* controllers[MAX_CONTROLLERS];
    SDL_Haptic* rumbles[MAX_CONTROLLERS];
};


class SDLEventInfo
{
public:
    SDL_Event event;
    bool first_in_queue = true;
    bool has_event = false;
};


// sdl_input.cpp
void process_controller_input(SDLControllerInput const& sdl, Input const& old_input, Input& new_input);

void process_keyboard_input(SDLEventInfo const& evt, KeyboardInput const& old_keyboard, KeyboardInput& new_keyboard);

void process_mouse_input(SDLEventInfo const& evt, MouseInput const& old_mouse, MouseInput& new_mouse);


constexpr u32 SCREEN_BYTES_PER_PIXEL = 4;

#ifndef SDL2_WASM

constexpr auto SDL_OPTIONS = SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER | SDL_INIT_HAPTIC;

#else

constexpr auto SDL_OPTIONS = SDL_INIT_VIDEO;

#endif


static void print_message(const char* msg)
{
#ifdef PRINT_MESSAGES
    printf("%s\n", msg);
#endif
}


static void print_sdl_error(const char* msg)
{
#ifdef PRINT_MESSAGES
    printf("%s\n%s\n", msg, SDL_GetError());
#endif
}


static void close_sdl()
{
    SDL_Quit();
}


static void display_error(const char* msg)
{
#ifndef SDL2_WASM
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "ERROR", msg, 0);
#endif

    print_sdl_error(msg);
}


static bool init_sdl()
{    
    if (SDL_Init(SDL_OPTIONS) != 0)
    {
        print_sdl_error("SDL_Init failed");
        return false;
    }

    return true;
}


static void handle_sdl_window_event(SDL_WindowEvent const& w_event)
{
#ifndef SDL2_WASM

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

#endif
}


static void open_game_controllers(SDLControllerInput& sdl, Input& input)
{
    int num_joysticks = SDL_NumJoysticks();
    int c = 0;
    for(int j = 0; j < num_joysticks; ++j)
    {
        if (!SDL_IsGameController(j))
        {
            continue;
        }

        print_message("found a controller");

        sdl.controllers[c] = SDL_GameControllerOpen(j);
        auto joystick = SDL_GameControllerGetJoystick(sdl.controllers[c]);
        if(!joystick)
        {
            print_message("no joystick");
        }

        sdl.rumbles[c] = SDL_HapticOpenFromJoystick(joystick);
        if(!sdl.rumbles[c])
        {
            print_message("no rumble from joystick");
        }
        else if(SDL_HapticRumbleInit(sdl.rumbles[c]) != 0)
        {
            print_sdl_error("SDL_HapticRumbleInit failed");
            SDL_HapticClose(sdl.rumbles[c]);
            sdl.rumbles[c] = 0;
        }
        else
        {
            print_message("found a rumble");
        }

        ++c;

        if (c >= MAX_CONTROLLERS)
        {
            break;
        }
    }

    input.num_controllers = c;
}


static void close_game_controllers(SDLControllerInput& sdl, Input const& input)
{
    for(u32 c = 0; c < input.num_controllers; ++c)
    {
        if(sdl.rumbles[c])
        {
            SDL_HapticClose(sdl.rumbles[c]);
        }
        SDL_GameControllerClose(sdl.controllers[c]);
    }
}


static void set_window_icon(SDL_Window* window)
{
    // this will "paste" the struct my_icon into this function
/*#include "../resources/icon_64.h"

// these masks are needed to tell SDL_CreateRGBSurface(From)
// to assume the data it gets is byte-wise RGB(A) data
    Uint32 rmask, gmask, bmask, amask;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    int shift = (window_icon.bytes_per_pixel == 3) ? 8 : 0;
    rmask = 0xff000000 >> shift;
    gmask = 0x00ff0000 >> shift;
    bmask = 0x0000ff00 >> shift;
    amask = 0x000000ff >> shift;
#else // little endian, like x86
    rmask = 0x000000ff;
    gmask = 0x0000ff00;
    bmask = 0x00ff0000;
    amask = (window_icon.bytes_per_pixel == 3) ? 0 : 0xff000000;
#endif

    SDL_Surface* icon = SDL_CreateRGBSurfaceFrom(
        (void*)window_icon.pixel_data,
        window_icon.width,
        window_icon.height,
        window_icon.bytes_per_pixel * 8,
        window_icon.bytes_per_pixel * window_icon.width,
        rmask, gmask, bmask, amask);

    SDL_SetWindowIcon(window, icon);

    SDL_FreeSurface(icon);*/
}


#ifndef SDL2_IMPL_B

class ScreenMemory
{
public:

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;

    void* image_data;
    int image_width;
    int image_height;
};


static void destroy_screen_memory(ScreenMemory& screen)
{
    if (screen.image_data)
    {
        free(screen.image_data);
    }

    if (screen.texture)
    {
        SDL_DestroyTexture(screen.texture);
    }

    if (screen.renderer)
    {
        SDL_DestroyRenderer(screen.renderer);
    }

    if(screen.window)
    {
        SDL_DestroyWindow(screen.window);
    }
}


static bool create_screen_memory(ScreenMemory& screen, const char* title, int width, int height)
{
    destroy_screen_memory(screen);

    screen.window = SDL_CreateWindow(
        title,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        width,
        height,
        SDL_WINDOW_RESIZABLE);

    if(!screen.window)
    {
        display_error("SDL_CreateWindow failed");
        return false;
    }

    set_window_icon(screen.window);

    screen.renderer = SDL_CreateRenderer(screen.window, -1, 0);

    if(!screen.renderer)
    {
        display_error("SDL_CreateRenderer failed");
        destroy_screen_memory(screen);
        return false;
    }

    screen.texture =  SDL_CreateTexture(
        screen.renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        width,
        height);
    
    if(!screen.texture)
    {
        display_error("SDL_CreateTexture failed");
        destroy_screen_memory(screen);
        return false;
    }

    screen.image_data = malloc((size_t)SCREEN_BYTES_PER_PIXEL * width * height);

    if(!screen.image_data)
    {
        display_error("Allocating image memory failed");
        destroy_screen_memory(screen);
        return false;
    }

    screen.image_width = width;
    screen.image_height = height;

    return true;
}


static void render_screen(ScreenMemory const& screen)
{
    auto const pitch = screen.image_width * SCREEN_BYTES_PER_PIXEL;
    auto error = SDL_UpdateTexture(screen.texture, 0, screen.image_data, pitch);
    if(error)
    {
        print_sdl_error("SDL_UpdateTexture failed");
    }

    error = SDL_RenderCopy(screen.renderer, screen.texture, 0, 0);
    if(error)
    {
        print_sdl_error("SDL_RenderCopy failed");
    }
    
    SDL_RenderPresent(screen.renderer);
}

#else



class ScreenMemory
{
public:

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
    SDL_Surface* surface = nullptr;

    void* image_data = nullptr;
    
    int image_width;
    int image_height;

    bool surface_locked = false;
};


static void lock_surface(ScreenMemory& screen)
{
    //assert(screen.surface);

    if (!screen.surface_locked && SDL_MUSTLOCK(screen.surface)) 
    {
        SDL_LockSurface(screen.surface);
        screen.surface_locked = true;
    }
}


static void unlock_surface(ScreenMemory& screen)
{
    //assert(screen.surface);

    if (screen.surface_locked && SDL_MUSTLOCK(screen.surface)) 
    {
        SDL_UnlockSurface(screen.surface);
        screen.surface_locked = false;
    }
}


static void destroy_screen_memory(ScreenMemory& screen)
{
    unlock_surface(screen);

    if (screen.texture)
    {
        SDL_DestroyTexture(screen.texture);
    }

    if (screen.renderer)
    {
        SDL_DestroyRenderer(screen.renderer);
    }

    if(screen.window)
    {
        SDL_DestroyWindow(screen.window);
    }
}


static bool create_screen_memory(ScreenMemory& screen, const char* title, int width, int height)
{
    destroy_screen_memory(screen);

    screen.window = SDL_CreateWindow(
        title,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        width,
        height,
        SDL_WINDOW_RESIZABLE);

    if(!screen.window)
    {
        display_error("SDL_CreateWindow failed");
        return false;
    }

    set_window_icon(screen.window);

    screen.renderer = SDL_CreateRenderer(screen.window, -1, 0);

    if(!screen.renderer)
    {
        display_error("SDL_CreateRenderer failed");
        destroy_screen_memory(screen);
        return false;
    }

    screen.surface = SDL_CreateRGBSurface(
        0,
        width,
        height,
        SCREEN_BYTES_PER_PIXEL * 8,
        0, 0, 0, 0);

    if(!screen.surface)
    {
        display_error("SDL_CreateRGBSurface failed");
        destroy_screen_memory(screen);
        return false;
    }

    screen.texture =  SDL_CreateTextureFromSurface(screen.renderer, screen.surface);    
    
    if(!screen.texture)
    {
        display_error("SDL_CreateTextureFromSurface");
        destroy_screen_memory(screen);
        return false;
    }    

    screen.image_data = (void*)(screen.surface->pixels);

    screen.image_width = width;
    screen.image_height = height;

    return true;
}


static void render_screen(ScreenMemory& screen)
{
    unlock_surface(screen);

    auto const pitch = screen.image_width * SCREEN_BYTES_PER_PIXEL;
    auto error = SDL_UpdateTexture(screen.texture, 0, screen.image_data, pitch);
    if(error)
    {
        print_sdl_error("SDL_UpdateTexture failed");
    }

    error = SDL_RenderCopy(screen.renderer, screen.texture, 0, 0);
    if(error)
    {
        print_sdl_error("SDL_RenderCopy failed");
    }
    
    SDL_RenderPresent(screen.renderer);
    
    lock_surface(screen);
}

#endif