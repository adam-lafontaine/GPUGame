#pragma once

#include "keyboard.hpp"
#include "mouse.hpp"
#include "controller.hpp"


constexpr u32 MAX_CONTROLLERS = 1;


class Input
{
public:
	KeyboardInput keyboard;
	MouseInput mouse;

	ControllerInput controllers[MAX_CONTROLLERS];
	u32 num_controllers;

	f32 dt_frame;
};
