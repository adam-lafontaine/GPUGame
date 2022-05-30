#include "app_types.hpp"

static bool make_device_assets(DeviceAssets& assets, device::MemoryBuffer& buffer)
{
    if(!make_device_tile(assets.grass_tile, buffer))
    {
        return false;
    }

    if(!make_device_tile(assets.brown_tile, buffer))
    {
        return false;
    }

    if(!make_device_tile(assets.black_tile, buffer))
    {
        return false;
    }

    return true;
}


static bool make_device_entity_array(DeviceEntityArray& array, device::MemoryBuffer& buffer, u32 n_elements)
{
    auto data_size = device_entity_array_data_size(n_elements);

    auto entity_data = device::push_bytes(buffer, data_size);
    if(!entity_data)
    {
        return false;
    }

    array.n_elements = n_elements;
    array.data = (Entity*)entity_data;

    return true;
}


static bool make_device_tile_matrix(DeviceTileMatrix& tilemap, device::MemoryBuffer& buffer, u32 width_tile, u32 height_tile)
{
    auto data_size = device_tile_matrix_data_size(width_tile * height_tile);

    auto tile_data = device::push_bytes(buffer, data_size);
    if(!tile_data)
    {
        return false;
    }

    tilemap.width = width_tile;
    tilemap.height = height_tile;
    tilemap.data = (DeviceTile*)tile_data;

    return true;
}


bool make_device_memory(DeviceMemory& memory, device::MemoryBuffer& buffer, u32 n_entities, u32 width_tile, u32 height_tile)
{
    if(!make_device_assets(memory.assets, buffer))
    {
        return false;
    }

    if(!make_device_entity_array(memory.entities, buffer, n_entities))
    {
        return false;
    }

    if(!make_device_tile_matrix(memory.tilemap, buffer, width_tile, height_tile))
    {
        return false;
    } 

    return true;
}


bool make_unified_memory(UnifiedMemory& memory, device::MemoryBuffer& buffer, u32 screen_width, u32 screen_height)
{
    if(!make_device_image(memory.screen_pixels, buffer, screen_width, screen_height))
    {
        return false;
    }

    if(!make_device_input_list(memory.previous_inputs, buffer))
    {
        return false;
    }

    if(!make_device_input_list(memory.current_inputs, buffer))
    {
        return false;
    }

    return true;
}