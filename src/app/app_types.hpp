#pragma once

#include "tiles/device_tile.hpp"
#include "input_record/input_record.hpp"


class WorldPosition
{
public:
    Point2Di32 tile;
    Point2Dr32 offset_m;
};


class Entity
{
public:
    r32 width;
    r32 height;
    Pixel color;

    WorldPosition position;
    Vec2Dr32 dt;
    r32 speed;

    Vec2Dr32 delta_pos_m;

    WorldPosition next_position;

    bool is_active = false;

    bool inv_x = false;
    bool inv_y = false;
};


using DeviceEntityArray = DeviceArray<Entity>;


constexpr size_t device_entity_array_data_size(u32 n_elements)
{
    return n_elements * sizeof(Entity);
}


inline bool make_device_entity_array(DeviceEntityArray& array, device::MemoryBuffer& buffer, u32 n_elements)
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


using DeviceTileMatrix = DeviceMatrix<DeviceTile>;


constexpr size_t device_tile_matrix_data_size(u32 n_tiles)
{
    return n_tiles * sizeof(DeviceTile);
}


inline bool make_device_tile_matrix(DeviceTileMatrix& tilemap, device::MemoryBuffer& buffer, u32 width_tile, u32 height_tile)
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


class DeviceAssets
{
public:
    DeviceTile grass_tile;
    
    DeviceTile brown_tile;
    DeviceTile black_tile;
};


constexpr size_t device_assets_data_size()
{
    return  N_TILE_BITMAPS * device_tile_data_size();
}


inline bool make_device_assets(DeviceAssets& assets, device::MemoryBuffer& buffer)
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



class DeviceMemory
{
public: 
    DeviceAssets assets;
    
    DeviceTileMatrix tilemap;

    DeviceArray<Entity> entities;    

};


constexpr size_t device_memory_total_size(u32 n_entities, u32 n_tiles)
{
    return 
        sizeof(DeviceMemory)
        + device_assets_data_size()
        + device_tile_matrix_data_size(n_tiles)
        + device_entity_array_data_size(n_entities);
}


inline bool make_device_memory(DeviceMemory& memory, device::MemoryBuffer& buffer, u32 n_entities, u32 width_tile, u32 height_tile)
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


class UnifiedMemory
{
public:

    DeviceImage screen_pixels;

    DeviceInputList previous_inputs;
    
    DeviceInputList current_inputs;
    
};


constexpr size_t unified_memory_total_size(u32 screen_width, u32 screen_height)
{
    return 
        sizeof(UnifiedMemory)
        + device_image_data_size(screen_width, screen_height)
        + device_input_list_data_size()
        + device_input_list_data_size();
}


inline bool make_unified_memory(UnifiedMemory& memory, device::MemoryBuffer& buffer, u32 screen_width, u32 screen_height)
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


using HostImage = Image;


class HostMemory
{
public:

    //HostImage screen_pixels;

    
};


class StateProps
{
public:

    u64 frame_count;
    bool reset_frame_count;

    u32 screen_width_px;
    u32 screen_height_px;

    r32 screen_width_m;

    WorldPosition screen_position;
};


class AppState
{
public:

    device::MemoryBuffer device_buffer;
    device::MemoryBuffer unified_buffer;

    DeviceMemory* device;
    UnifiedMemory* unified;

    HostMemory host;
    StateProps props;

    DeviceAssets device_assets;
};
