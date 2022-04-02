#pragma once

#include "../app_types.hpp"
#include "gpu_constants.hpp"


namespace gpu
{
    void init_device_memory(DeviceMemory const& device, DeviceBuffer const& not_buffer);

    void update(AppState& state);

    void render(AppState& state);
}