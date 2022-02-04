#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using r32 = float;
using r64 = double;

using b32 = uint32_t;

#define ArrayCount(arr) (sizeof(arr) / sizeof((arr)[0]))
#define Kilobytes(value) ((value) * 1024LL)
#define Megabytes(value) (Kilobytes(value) * 1024LL)
#define Gigabytes(value) (Megabytes(value) * 1024LL)
#define Terabytes(value) (Gigabytes(value) * 1024LL)

#define GlobalVariable static

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


class Vec2Dr64
{
public:
	r64 x;
	r64 y;
};


class Rect2Du32
{
public:
	u32 x_begin;
	u32 x_end;
	u32 y_begin;
	u32 y_end;
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
using Point2Dr64 = Vec2Dr64;
using Point2Du32 = Vec2Du32;

//#define NO_CPP_17
//#define NDEBUG


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