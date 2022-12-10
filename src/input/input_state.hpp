#include "input.hpp"

inline void record_button_input(ButtonState const& old_state, ButtonState& new_state, b32 is_down)
{
    new_state.pressed = !old_state.is_down && is_down;
    new_state.is_down = is_down;
	new_state.raised = old_state.is_down && !is_down;
}


inline void reset_button_state(ButtonState& state)
{
	for (u32 i = 0; i < ArrayCount(state.states); ++i)
	{
		state.states[i] = false;
	}
}


inline void copy_button_state(ButtonState const& src, ButtonState& dst)
{
	for (u32 i = 0; i < ArrayCount(src.states); ++i)
	{
		dst.is_down = src.is_down;
		dst.pressed = false;
		dst.raised = false;
	}
}


inline void copy_controller_state(ControllerInput const& src, ControllerInput& dst)
{
	for (u32 i = 0; i < ArrayCount(src.buttons); ++i)
	{
		copy_button_state(src.buttons[i], dst.buttons[i]);
	}
}


inline void copy_keyboard_state(KeyboardInput const& src, KeyboardInput& dst)
{
	for (u32 i = 0; i < ArrayCount(src.keys); ++i)
	{
		copy_button_state(src.keys[i], dst.keys[i]);
	}
}


inline void copy_mouse_state(MouseInput const& src, MouseInput& dst)
{
	dst.win_pos.x = src.win_pos.x;
	dst.win_pos.y = src.win_pos.y;

	for (u32 i = 0; i < ArrayCount(src.buttons); ++i)
	{
		copy_button_state(src.buttons[i], dst.buttons[i]);
	}
}


inline void reset_mouse(MouseInput& mouse)
{
	mouse.win_pos.x = 0;
	mouse.win_pos.y = 0;

	for (u32 i = 0; i < ArrayCount(mouse.buttons); ++i)
	{
		reset_button_state(mouse.buttons[i]);
	}
}