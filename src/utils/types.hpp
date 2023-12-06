#pragma once

#include "defs.hpp"


class Vec2Di32
{
public:
    i32 x;
    i32 y;
};


class Vec2Dr32
{
public:
    r32 x;
    r32 y;
};


class Vec2Du32
{
public:
	u32 x;
	u32 y;
};


class Vec2Df64
{
public:
	f64 x;
	f64 y;
};


class Rect2Du32
{
public:
	u32 x_begin;
	u32 x_end;
	u32 y_begin;
	u32 y_end;
};


class Rect2Di32
{
public:
	i32 x_begin;
	i32 x_end;
	i32 y_begin;
	i32 y_end;
};


class Rect2Dr32
{
public:
	r32 x_begin;
	r32 x_end;
	r32 y_begin;
	r32 y_end;
};



using Range2Du32 = Rect2Du32;

using Point2Di32 = Vec2Di32;
using Point2Dr32 = Vec2Dr32;
using Point2Df64 = Vec2Df64;
using Point2Du32 = Vec2Du32;




template <typename T>
class Matrix
{
public:
	u32 width;
	u32 height;

	T* data;
};


template <typename T>
class Array
{
public:
	u32 n_elements;
	T* data;
};


inline i32 round_r32_to_i32(r32 value)
{
    return (i32)roundf(value);
}


inline i32 round_r32_to_u32(r32 value)
{
    return (u32)(value + 0.5f);
}


inline u8 scale_r32_to_u8(r32 value)
{
    if(value < 0.0f)
        return 0;

    if(value > 255.0f)
        return 255;

    return (u8)round_r32_to_u32(value);
}


inline i32 floor_r32_to_i32(r32 value)
{
    return (i32)(floorf(value));
}

inline r32 absolute_value(r32 value)
{
    return fabsf(value);
}


inline r32 square_root(r32 value)
{
    return sqrtf(value);
}