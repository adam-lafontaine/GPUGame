#pragma once

#include "../input/input.hpp"

#include <SDL.h>

class SDLInput
{
public:
    SDL_GameController* controllers[MAX_CONTROLLERS];
    SDL_Haptic* rumbles[MAX_CONTROLLERS];
};


void process_controller_input(SDLInput const& sdl, Input const& old_input, Input& new_input);

void process_keyboard_input(b32 has_event, SDL_Event const& event, Input const& old_input, Input& new_input);

void process_mouse_input(b32 has_event, SDL_Event const& event, Input const& old_input, Input& new_input);