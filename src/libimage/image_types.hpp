#pragma once

#include "../utils/defs.hpp"


constexpr auto RGB_CHANNELS = 3u;
constexpr auto RGBA_CHANNELS = 4u;


typedef union pixel_t
{
	struct
	{
        u8 blue;
		u8 green;
		u8 red;
		u8 alpha;		
	};

	u8 channels[RGBA_CHANNELS];

	u32 value;

} Pixel;


class Image
{
public:
    u32 width;
    u32 height;

    Pixel* data;
};


inline pixel_t to_pixel(u8 red, u8 green, u8 blue)
{
	pixel_t p{};

	p.alpha = 255;
	p.red = red;
	p.green = green;
	p.blue = blue;

	return p;
}