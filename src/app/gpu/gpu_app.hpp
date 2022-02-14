#pragma once

#include "../app_types.hpp"
#include "gpu_constants.hpp"


namespace gpu
{
    void init_device_memory(DeviceMemory const& device);

    void update(AppState& state);

    void render(AppState& state);
}

