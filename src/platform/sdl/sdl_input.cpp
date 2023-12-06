#include "sdl_include.hpp"
#include "../../input/input_state.hpp"


static f32 normalize_axis_value(f32 axis)
{
	f32 norm = axis / 32768.0f;
	if(norm > 1.0f)
	{
		return 1.0f;
	}

	if(norm < -1.0f)
	{
		return -1.0f;
	}

	return norm;
}


static void record_axis_input(AxisState const& old_state, AxisState& new_state, f32 value)
{
	new_state.start = old_state.end;
	new_state.end = normalize_axis_value(value);

	// not correct?
	new_state.min = new_state.end;
	new_state.max = new_state.end;
}


static void record_controller_input(SDL_GameController* sdl, ControllerInput const& old_controller, ControllerInput& new_controller)
{
    copy_controller_state(old_controller, new_controller);

    if(!sdl || !SDL_GameControllerGetAttached(sdl))
    {
        return;
    }

	bool is_down = false;
	Sint16 axis = 0;

#if CONTROLLER_UP
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_DPAD_UP);
	record_button_input(old_controller.dpad_up, new_controller.dpad_up, is_down);
#endif
#if CONTROLLER_DOWN
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_DPAD_DOWN);
	record_button_input(old_controller.dpad_down, new_controller.dpad_down, is_down);
#endif
#if CONTROLLER_LEFT
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_DPAD_LEFT);
	record_button_input(old_controller.dpad_left, new_controller.dpad_left, is_down);
#endif
#if CONTROLLER_RIGHT
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_DPAD_RIGHT);
	record_button_input(old_controller.dpad_right, new_controller.dpad_right, is_down);
#endif
#if CONTROLLER_START
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_START);
	record_button_input(old_controller.button_start, new_controller.button_start, is_down);
#endif
#if CONTROLLER_BACK
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_BACK);
	record_button_input(old_controller.button_back, new_controller.button_back, is_down);
#endif
#if CONTROLLER_LEFT_SHOULDER
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_LEFTSHOULDER);
	record_button_input(old_controller.shoulder_left, new_controller.shoulder_left, is_down);
#endif
#if CONTROLLER_RIGHT_SHOULDER
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER);
	record_button_input(old_controller.shoulder_right, new_controller.shoulder_right, is_down);
#endif
#if CONTROLLER_A
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_A);
	record_button_input(old_controller.button_a, new_controller.button_a, is_down);
#endif
#if CONTROLLER_B
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_B);
	record_button_input(old_controller.button_b, new_controller.button_b, is_down);
#endif
#if CONTROLLER_X
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_X);
	record_button_input(old_controller.button_x, new_controller.button_x, is_down);
#endif
#if CONTROLLER_Y
	is_down = SDL_GameControllerGetButton(sdl, SDL_CONTROLLER_BUTTON_Y);
	record_button_input(old_controller.button_y, new_controller.button_y, is_down);
#endif
#if CONTROLLER_STICK_LEFT
	axis = SDL_GameControllerGetAxis(sdl, SDL_CONTROLLER_AXIS_LEFTX);
	record_axis_input(old_controller.stick_left_x, new_controller.stick_left_x, axis);

	axis = SDL_GameControllerGetAxis(sdl, SDL_CONTROLLER_AXIS_LEFTY);
	record_axis_input(old_controller.stick_left_y, new_controller.stick_left_y, -1.0f * axis);
#endif
#if CONTROLLER_STICK_RIGHT
	axis = SDL_GameControllerGetAxis(sdl, SDL_CONTROLLER_AXIS_RIGHTX);
	record_axis_input(old_controller.stick_right_x, new_controller.stick_right_x, axis);

	axis = SDL_GameControllerGetAxis(sdl, SDL_CONTROLLER_AXIS_RIGHTY);
	record_axis_input(old_controller.stick_right_y, new_controller.stick_right_y, -1.0f * axis);
#endif
#if CONTROLLER_TRIGGER_LEFT
	axis = SDL_GameControllerGetAxis(sdl, SDL_CONTROLLER_AXIS_TRIGGERLEFT);
	record_axis_input(old_controller.trigger_left, new_controller.trigger_left, axis);
#endif
#if CONTROLLER_TRIGGER_RIGHT
	axis = SDL_GameControllerGetAxis(sdl, SDL_CONTROLLER_AXIS_TRIGGERRIGHT);
	record_axis_input(old_controller.trigger_right, new_controller.trigger_right, axis);
#endif
}


void process_controller_input(SDLControllerInput const& sdl, Input const& old_input, Input& new_input)
{
    for(u32 c = 0; c < new_input.num_controllers; ++c)
    {
        record_controller_input(sdl.controllers[c], old_input.controllers[c], new_input.controllers[c]);
    }
}


static void record_keyboard_input(SDL_Keycode key_code, KeyboardInput const& old_input, KeyboardInput& new_input, bool is_down)
{
	switch (key_code)
	{
#if KEYBOARD_A
	case SDLK_a:
		record_button_input(old_input.a_key, new_input.a_key, is_down);
		break;
#endif
#if KEYBOARD_B
	case SDLK_b:
		record_button_input(old_input.b_key, new_input.b_key, is_down);
		break;
#endif
#if KEYBOARD_C
	case SDLK_c:
		record_button_input(old_input.c_key, new_input.c_key, is_down);
		break;
#endif
#if KEYBOARD_D
	case SDLK_d:
		record_button_input(old_input.d_key, new_input.d_key, is_down);
		break;
#endif
#if KEYBOARD_E
	case SDLK_e:
		record_button_input(old_input.e_key, new_input.e_key, is_down);
		break;
#endif
#if KEYBOARD_F
	case SDLK_f:
		record_button_input(old_input.f_key, new_input.f_key, is_down);
		break;
#endif
#if KEYBOARD_G
	SDLK_g:
		record_button_input(old_input.g_key, new_input.g_key, is_down);
		break;
#endif
#if KEYBOARD_H
	case SDLK_h:
		record_button_input(old_input.h_key, new_input.h_key, is_down);
		break;
#endif
#if KEYBOARD_I
	case SDLK_i:
		record_button_input(old_input.i_key, new_input.i_key, is_down);
		break;
#endif
#if KEYBOARD_J
	case SDLK_j:
		record_button_input(old_input.j_key, new_input.j_key, is_down);
		break;
#endif
#if KEYBOARD_K
	case SDLK_k:
		record_button_input(old_input.k_key, new_input.k_key, is_down);
		break;
#endif
#if KEYBOARD_L
	case SDLK_l:
		record_button_input(old_input.l_key, new_input.l_key, is_down);
		break;
#endif
#if KEYBOARD_M
	case SDLK_m:
		record_button_input(old_input.m_key, new_input.m_key, is_down);
		break;
#endif
#if KEYBOARD_N
	case SDLK_n:
		record_button_input(old_input.n_key, new_input.n_key, is_down);
		break;
#endif
#if KEYBOARD_O
	case SDLK_o:
		record_button_input(old_input.o_key, new_input.o_key, is_down);
		break;
#endif
#if KEYBOARD_P
	case SDLK_p:
		record_button_input(old_input.p_key, new_input.p_key, is_down);
		break;
#endif
#if KEYBOARD_Q
	case SDLK_q:
		record_button_input(old_input.q_key, new_input.q_key, is_down);
		break;
#endif
#if KEYBOARD_R
	case SDLK_r:
		record_button_input(old_input.r_key, new_input.r_key, is_down);
		break;
#endif
#if KEYBOARD_S
	case SDLK_s:
		record_button_input(old_input.s_key, new_input.s_key, is_down);
		break;
#endif
#if KEYBOARD_T
	case SDLK_t:
		record_button_input(old_input.t_key, new_input.t_key, is_down);
		break;
#endif
#if KEYBOARD_U
	case SDLK_u:
		record_button_input(old_input.u_key, new_input.u_key, is_down);
		break;
#endif
#if KEYBOARD_V
	case SDLK_v:
		record_button_input(old_input.v_key, new_input.v_key, is_down);
		break;
#endif
#if KEYBOARD_W
	case SDLK_w:
		record_button_input(old_input.w_key, new_input.w_key, is_down);
		break;
#endif
#if KEYBOARD_X
	case SDLK_x:
		record_button_input(old_input.x_key, new_input.x_key, is_down);
		break;
#endif
#if KEYBOARD_Y
	case SDLK_y:
		record_button_input(old_input.y_key, new_input.y_key, is_down);
		break;
#endif
#if KEYBOARD_Z
	case SDLK_z:
		record_button_input(old_input.z_key, new_input.z_key, is_down);
		break;
#endif
#if KEYBOARD_0
	case SDLK_0:
		record_button_input(old_input.z_key, new_input.z_key, is_down);
		break;
#endif
#if KEYBOARD_1
	case SDLK_1:
		record_button_input(old_input.one_key, new_input.one_key, is_down);
		break;
#endif
#if KEYBOARD_2
	case SDLK_2:
		record_button_input(old_input.two_key, new_input.two_key, is_down);
		break;
#endif
#if KEYBOARD_3
	case SDLK_3:
		record_button_input(old_input.three_key, new_input.three_key, is_down);
		break;
#endif
#if KEYBOARD_4
	case SDLK_4:
		record_button_input(old_input.four_key, new_input.four_key, is_down);
		break;
#endif
#if KEYBOARD_5
	case SDLK_5:
		record_button_input(old_input.five_key, new_input.five_key, is_down);
		break;
#endif
#if KEYBOARD_6
	case SDLK_6:
		record_button_input(old_input.six_key, new_input.six_key, is_down);
		break;
#endif
#if KEYBOARD_7
	case SDLK_7:
		record_button_input(old_input.seven_key, new_input.seven_key, is_down);
		break;
#endif
#if KEYBOARD_8
	case SDLK_8:
		record_button_input(old_input.eight_key, new_input.eight_key, is_down);
		break;
#endif
#if KEYBOARD_9
	case SDLK_9:
		record_button_input(old_input.nine_key, new_input.nine_key, is_down);
		break;
#endif
#if KEYBOARD_UP
	case SDLK_UP:
		record_button_input(old_input.up_key, new_input.up_key, is_down);
		break;
#endif
#if KEYBOARD_DOWN
	case SDLK_DOWN:
		record_button_input(old_input.down_key, new_input.down_key, is_down);
		break;
#endif
#if KEYBOARD_LEFT
	case SDLK_LEFT:
		record_button_input(old_input.left_key, new_input.left_key, is_down);
		break;
#endif
#if KEYBOARD_RIGHT
	case SDLK_RIGHT:
		record_button_input(old_input.right_key, new_input.right_key, is_down);
		break;
#endif
#if KEYBOARD_RETURN
	case SDLK_RETURN:
		record_button_input(old_input.return_key, new_input.return_key, is_down);
		break;
#endif
#if KEYBOARD_ESCAPE
	case SDLK_ESCAPE:
		record_button_input(old_input.escape_key, new_input.escape_key, is_down);
		break;
#endif
#if KEYBOARD_SPACE
	case SDLK_SPACE:
		record_button_input(old_input.space_key, new_input.space_key, is_down);
		break;
#endif
#if KEYBOARD_SHIFT
		case SDLK_LSHIFT:
			record_button_input(old_input.shift_key, new_input.shift_key, is_down);
			break;
		case SDLK_RSHIFT:
			record_button_input(old_input.shift_key, new_input.shift_key, is_down);
			break;
#endif
#if KEYBOARD_PLUS
	case SDLK_KP_PLUS:
		record_button_input(old_input.plus_key, new_input.plus_key, is_down);
		break;
#endif
#if KEYBOARD_MINUS
	case SDLK_KP_MINUS:
		record_button_input(old_input.minus_key, new_input.minus_key, is_down);
		break;
#endif
#if KEYBOARD_MULTIPLY
	case SDLK_KP_MULTIPLY:
		record_button_input(old_input.mult_key, new_input.mult_key, is_down);
		break;
#endif
#if KEYBOARD_DIVIDE
	case SDLK_KP_DIVIDE:
		record_button_input(old_input.div_key, new_input.div_key, is_down);
		break;
#endif
#if KEYBOARD_NUMPAD_0
		case SDLK_KP_0:
			record_button_input(old_input.np_zero_key, new_input.np_zero_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_1
		case SDLK_KP_1:
			record_button_input(old_input.np_one_key, new_input.np_one_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_2
		case SDLK_KP_2:
			record_button_input(old_input.np_two_key, new_input.np_two_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_3
		case SDLK_KP_3:
			record_button_input(old_input.np_three_key, new_input.np_three_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_4
		case SDLK_KP_4:
			record_button_input(old_input.np_four_key, new_input.np_four_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_5
		case SDLK_KP_5:
			record_button_input(old_input.np_five_key, new_input.np_five_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_6
		case SDLK_KP_6:
			record_button_input(old_input.np_six_key, new_input.np_six_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_7
		case SDLK_KP_7:
			record_button_input(old_input.np_seven_key, new_input.np_seven_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_8
		case SDLK_KP_8:
			record_button_input(old_input.np_eight_key, new_input.np_eight_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_9
		case SDLK_KP_9:
			record_button_input(old_input.np_nine_key, new_input.np_nine_key, is_down);
			break;
#endif
	
	default:
		break;
	}
}


void process_keyboard_input(SDLEventInfo const& evt, KeyboardInput const& old_keyboard, KeyboardInput& new_keyboard)
{
	if (evt.first_in_queue)
	{
		copy_keyboard_state(old_keyboard, new_keyboard);
	}

	if (!evt.has_event)
	{
		return;
	}

	auto event = evt.event;

	switch (event.type)
	{
	case SDL_KEYDOWN:
	case SDL_KEYUP:
	{
		if (event.key.repeat)
		{
			return;
		}

		bool is_down = event.key.state == SDL_PRESSED;

		auto key_code = event.key.keysym.sym;
		record_keyboard_input(key_code, old_keyboard, new_keyboard, is_down);
	} break;
	}
}


static void record_mouse_button_input(Uint8 button_code, MouseInput const& old_input, MouseInput& new_input, bool is_down)
{
	switch(button_code)
	{		
#if MOUSE_LEFT
		case SDL_BUTTON_LEFT:
		{
			record_button_input(old_input.button_left, new_input.button_left, is_down);
		} break;
#endif
#if MOUSE_RIGHT
		case SDL_BUTTON_RIGHT:
		{
			record_button_input(old_input.button_right, new_input.button_right, is_down);
		} break;
#endif
#if MOUSE_MIDDLE
		case SDL_BUTTON_MIDDLE:
		{
			record_button_input(old_input.button_middle, new_input.button_middle, is_down);
		} break;
#endif
#if MOUSE_X1
		case SDL_BUTTON_X1:
		{
			record_button_input(old_input.button_x1, new_input.button_x1, is_down);
		} break;
#endif
#if MOUSE_X2
		case SDL_BUTTON_X2:
		{
			record_button_input(old_input.button_x1, new_input.button_x1, is_down);
		} break;
#endif
	}
}


void process_mouse_input(SDLEventInfo const& evt, MouseInput const& old_mouse, MouseInput& new_mouse)
{
	if (evt.first_in_queue)
	{
		copy_mouse_state(old_mouse, new_mouse);
	}

	if (!evt.has_event)
	{
		return;
	}

	auto event = evt.event;

	auto& mouse = new_mouse;

	switch (event.type)
	{
	case SDL_MOUSEBUTTONDOWN:
	case SDL_MOUSEBUTTONUP:
	{
		bool is_down = event.type == SDL_MOUSEBUTTONDOWN;
		auto button = event.button.button;

		record_mouse_button_input(button, old_mouse, mouse, is_down);
	} break;
	case SDL_MOUSEMOTION:
	{
		auto& motion = event.motion;

		mouse.win_pos.x = motion.x;
		mouse.win_pos.y = motion.y;

	} break;
	case SDL_MOUSEWHEEL:
	{

	} break;
	}
}