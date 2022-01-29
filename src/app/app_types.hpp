#pragma once

#include "../device/device.hpp"


class DeviceMemory
{
public:
    DeviceBuffer buffer;
    
};


class UnifiedMemory
{
public:
    DeviceBuffer buffer;

    DeviceImage screen_pixels;
};


class AppState
{
public:
    
    DeviceMemory device;
    UnifiedMemory unified;
};
