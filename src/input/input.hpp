#pragma once

#include "keyboard.hpp"
#include "mouse.hpp"
#include "controller.hpp"


constexpr u32 MAX_CONTROLLERS = 1;


typedef struct input_t
{
	KeyboardInput keyboard;
	MouseInput mouse;

	ControllerInput controllers[MAX_CONTROLLERS];
	u32 num_controllers;

	r32 dt_frame;

} Input;
