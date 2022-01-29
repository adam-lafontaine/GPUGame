#pragma once

#include <cstdint>
#include <cstddef>

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


class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;
	u32 y_begin;
	u32 y_end;
};


using Point2Di32 = Vec2Di32;
using Point2Dr32 = Vec2Di32;
using Point2Dr64 = Vec2Dr64;
using Point2Du32 = Vec2Du32;

//#define NO_CPP_17
//#define NDEBUG