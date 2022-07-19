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
    constexpr auto n_tile_bitmaps = sizeof(DeviceAssets) / sizeof(DeviceTile);

    return  n_tile_bitmaps * device_tile_data_size();
}





static bool make_device_assets(DeviceAssets& assets, device::DeviceBuffer& buffer)
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


static bool make_device_entity_array(DeviceEntityArray& array, device::DeviceBuffer& buffer, u32 n_elements)
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


static bool make_device_tile_matrix(DeviceTileMatrix& tilemap, device::DeviceBuffer& buffer, u32 width_tile, u32 height_tile)
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
        sizeof(DeviceMemoryOld)
        + device_assets_data_size()
        + device_tile_matrix_data_size(WORLD_WIDTH_TILE * WORLD_HEIGHT_TILE)
        + device_entity_array_data_size(N_BLUE_ENTITIES)
        + device_entity_array_data_size(N_BROWN_ENTITIES)
        + device_entity_array_data_size(5);
}


bool make_device_memory(DeviceMemoryOld& memory, device::DeviceBuffer& buffer)
{

    if(!make_device_tile_matrix(memory.tilemap_old, buffer, WORLD_WIDTH_TILE, WORLD_HEIGHT_TILE))
    {
        return false;
    }    
    
    if(!make_device_assets(memory.assets_old, buffer))
    {
        return false;
    }

    if(!make_device_entity_array(memory.wall_entities_old, buffer, N_BROWN_ENTITIES))
    {
        return false;
    }

    if(!make_device_entity_array(memory.blue_entities_old, buffer, N_BLUE_ENTITIES))
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