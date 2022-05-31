#include "app_types.hpp"
#include "app_include.hpp"


static size_t device_entity_array_data_size(u32 n_elements)
{
    return n_elements * sizeof(Entity);
}


static size_t device_tile_matrix_data_size(u32 n_tiles)
{
    return n_tiles * sizeof(DeviceTile);
}


static size_t device_assets_data_size()
{
    return  N_TILE_BITMAPS * device_tile_data_size();
}


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


size_t device_memory_total_size()
{
    return 
        sizeof(DeviceMemory)
        + device_assets_data_size()
        + device_tile_matrix_data_size(WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE)
        + device_entity_array_data_size(N_BLUE_ENTITIES)
        + device_entity_array_data_size(N_BROWN_ENTITIES)
        + device_entity_array_data_size(1);
}


bool make_device_memory(DeviceMemory& memory, device::MemoryBuffer& buffer)
{

    if(!make_device_tile_matrix(memory.tilemap, buffer, WORLD_WIDTH_TILE, WORLD_HEIGHT_TILE))
    {
        return false;
    }
    
    
    if(!make_device_assets(memory.assets, buffer))
    {
        return false;
    }

    if(!make_device_entity_array(memory.wall_entities, buffer, N_BROWN_ENTITIES))
    {
        return false;
    }

    if(!make_device_entity_array(memory.blue_entities, buffer, N_BLUE_ENTITIES))
    {
        return false;
    }
    
    // will fail to run if this not here
    if(!make_device_entity_array(memory.memory_bug, buffer, 1))
    {
        return false;
    }
    

    return true;
}


size_t unified_memory_total_size(u32 screen_width, u32 screen_height)
{
    return 
        sizeof(UnifiedMemory)
        + device_image_data_size(screen_width, screen_height)
        + device_input_list_data_size()
        + device_input_list_data_size();
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