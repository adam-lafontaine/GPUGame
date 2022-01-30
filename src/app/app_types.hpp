#pragma once

#include "../device/device.hpp"


class DeviceMemory
{
public:
    DeviceBuffer buffer;

    DeviceArray<r32> r32_array; // just because
    
};


class UnifiedMemory
{
public:
    DeviceBuffer buffer;

    DeviceImage screen_pixels;
};


class HostMemory
{
public:
    u32* elements; // just because
    u32 n_elements;
};


class StateProps
{
public:
    u8 red; // just because
    u8 green;
    u8 blue;
};


class AppState
{
public:
    
    DeviceMemory device;
    UnifiedMemory unified;

    HostMemory host;

    StateProps props;    
};


inline void init_state_props(AppState& state)
{
    state.props.red = 55;
    state.props.green = 155;
    state.props.blue = 255;
}
