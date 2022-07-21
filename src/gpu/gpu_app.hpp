#pragma once

#include "../app/app_types.hpp"
#include "gpu_constants.hpp"


namespace gpu
{
    bool init_device_memory(AppState const& state);

    void update(AppState& state);

    void render(AppState& state);
}