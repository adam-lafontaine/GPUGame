#pragma once

#include "button_state.hpp"

// activate keys to accept input from
#define KEYBOARD_A 0
#define KEYBOARD_B 0
#define KEYBOARD_C 0
#define KEYBOARD_D 0
#define KEYBOARD_E 0
#define KEYBOARD_F 0
#define KEYBOARD_G 0
#define KEYBOARD_H 0
#define KEYBOARD_I 0
#define KEYBOARD_J 0
#define KEYBOARD_K 0
#define KEYBOARD_L 0
#define KEYBOARD_M 0
#define KEYBOARD_N 0
#define KEYBOARD_O 0
#define KEYBOARD_P 0
#define KEYBOARD_Q 0
#define KEYBOARD_R 0
#define KEYBOARD_S 0
#define KEYBOARD_T 0
#define KEYBOARD_U 0
#define KEYBOARD_V 0
#define KEYBOARD_W 0
#define KEYBOARD_X 0
#define KEYBOARD_Y 0
#define KEYBOARD_Z 0
#define KEYBOARD_0 0
#define KEYBOARD_1 0
#define KEYBOARD_2 0
#define KEYBOARD_3 0
#define KEYBOARD_4 0
#define KEYBOARD_5 0
#define KEYBOARD_6 0
#define KEYBOARD_7 0
#define KEYBOARD_8 0
#define KEYBOARD_9 0
#define KEYBOARD_UP 1
#define KEYBOARD_DOWN 1
#define KEYBOARD_LEFT 1
#define KEYBOARD_RIGHT 1
#define KEYBOARD_RETURN 0
#define KEYBOARD_ESCAPE 0
#define KEYBOARD_SPACE 1
#define KEYBOARD_PLUS 0
#define KEYBOARD_MINUS 0
#define KEYBOARD_MULTIPLY 0
#define KEYBOARD_DIVIDE 0


constexpr size_t KEYBOARD_KEYS = 
KEYBOARD_A 
+ KEYBOARD_B 
+ KEYBOARD_C
+ KEYBOARD_D
+ KEYBOARD_E
+ KEYBOARD_F
+ KEYBOARD_G
+ KEYBOARD_H
+ KEYBOARD_I
+ KEYBOARD_J
+ KEYBOARD_K
+ KEYBOARD_L
+ KEYBOARD_M
+ KEYBOARD_N
+ KEYBOARD_O
+ KEYBOARD_P
+ KEYBOARD_Q
+ KEYBOARD_R
+ KEYBOARD_S
+ KEYBOARD_T
+ KEYBOARD_U
+ KEYBOARD_V
+ KEYBOARD_W
+ KEYBOARD_X
+ KEYBOARD_Y
+ KEYBOARD_Z
+ KEYBOARD_0
+ KEYBOARD_1
+ KEYBOARD_2
+ KEYBOARD_3
+ KEYBOARD_4
+ KEYBOARD_5
+ KEYBOARD_6
+ KEYBOARD_7
+ KEYBOARD_8
+ KEYBOARD_9
+ KEYBOARD_UP
+ KEYBOARD_DOWN
+ KEYBOARD_LEFT
+ KEYBOARD_RIGHT
+ KEYBOARD_RETURN
+ KEYBOARD_ESCAPE
+ KEYBOARD_SPACE
+ KEYBOARD_PLUS
+ KEYBOARD_MINUS
+ KEYBOARD_MULTIPLY
+ KEYBOARD_DIVIDE
;


typedef union keyboard_input_t
{
	ButtonState keys[KEYBOARD_KEYS];
	struct
	{

#if KEYBOARD_A
		ButtonState a_key;
#endif
#if KEYBOARD_B
		ButtonState b_key;
#endif
#if KEYBOARD_C
		ButtonState c_key;
#endif
#if KEYBOARD_D
		ButtonState d_key;
#endif
#if KEYBOARD_E
		ButtonState e_key;
#endif
#if KEYBOARD_F
		ButtonState f_key;
#endif
#if KEYBOARD_G
		ButtonState g_key;
#endif
#if KEYBOARD_H
		ButtonState h_key;
#endif
#if KEYBOARD_I
		ButtonState i_key;
#endif
#if KEYBOARD_J
		ButtonState j_key;
#endif
#if KEYBOARD_K
		ButtonState k_key;
#endif
#if KEYBOARD_L
		ButtonState l_key;
#endif
#if KEYBOARD_M
		ButtonState m_key;
#endif
#if KEYBOARD_N
		ButtonState n_key;
#endif
#if KEYBOARD_O
		ButtonState o_key;
#endif
#if KEYBOARD_P
		ButtonState p_key;
#endif
#if KEYBOARD_Q
		ButtonState q_key;
#endif
#if KEYBOARD_R
		ButtonState r_key;
#endif
#if KEYBOARD_S
		ButtonState s_key;
#endif
#if KEYBOARD_T
		ButtonState t_key;
#endif
#if KEYBOARD_U
		ButtonState u_key;
#endif
#if KEYBOARD_V
		ButtonState v_key;
#endif
#if KEYBOARD_W
		ButtonState w_key;
#endif
#if KEYBOARD_X
		ButtonState x_key;
#endif
#if KEYBOARD_Y
		ButtonState y_key;
#endif
#if KEYBOARD_Z
		ButtonState z_key;
#endif
#if KEYBOARD_0
		ButtonState zero_key;
#endif
#if KEYBOARD_1
		ButtonState one_key;
#endif
#if KEYBOARD_2
		ButtonState two_key;
#endif
#if KEYBOARD_3
		ButtonState three_key;
#endif
#if KEYBOARD_4
		ButtonState four_key;
#endif
#if KEYBOARD_5
		ButtonState five_key;
#endif
#if KEYBOARD_6
		ButtonState six_key;
#endif
#if KEYBOARD_7
		ButtonState seven_key;
#endif
#if KEYBOARD_8
		ButtonState eight_key;
#endif
#if KEYBOARD_9
		ButtonState nine_key;
#endif
#if KEYBOARD_UP
		ButtonState up_key;
#endif
#if KEYBOARD_DOWN
		ButtonState down_key;
#endif
#if KEYBOARD_LEFT
		ButtonState left_key;
#endif
#if KEYBOARD_RIGHT
		ButtonState right_key;
#endif
#if KEYBOARD_RETURN
		ButtonState return_key;
#endif
#if KEYBOARD_ESCAPE
		ButtonState escape_key;
#endif
#if KEYBOARD_SPACE
		ButtonState space_key;
#endif
#if KEYBOARD_PLUS
		ButtonState plus_key;
#endif
#if KEYBOARD_MINUS
		ButtonState minus_key;
#endif
#if KEYBOARD_MULTIPLY
		ButtonState mult_key;
#endif
#if KEYBOARD_DIVIDE
		ButtonState div_key;
#endif

	};

} KeyboardInput;
