#pragma once

#include "../utils/types.hpp"

typedef union button_state_t
{
	b32 states[3];
	struct
	{
		b32 pressed;
		b32 is_down;
		b32 raised;
	};


} ButtonState;
