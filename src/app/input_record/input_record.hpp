#pragma once

#include "../../device/device.hpp"



using uInput = u32;

namespace INPUT
{
    constexpr uInput PLAYER_UP    = 0b0000'0000'0000'0000'0000'0000'0000'0001;
    constexpr uInput PLAYER_DOWN  = 0b0000'0000'0000'0000'0000'0000'0000'0010;
    constexpr uInput PLAYER_LEFT  = 0b0000'0000'0000'0000'0000'0000'0000'0100;
    constexpr uInput PLAYER_RIGHT = 0b0000'0000'0000'0000'0000'0000'0000'1000;

    constexpr u32 MAX_RECORDS = 5000;
}




class InputRecord
{
public:
    u64 frame_begin;
    u64 frame_end;
    uInput input;

    r32 est_dt_frame;
};


class DeviceInputList
{
public:
    u32 capacity;
    u32 size;
    u32 read_index;

    InputRecord* data;
};


constexpr size_t device_input_list_data_size()
{
    return sizeof(InputRecord) * INPUT::MAX_RECORDS;
}


inline bool make_device_input_list(DeviceInputList& list, device::MemoryBuffer& buffer)
{
    auto data_size = device_input_list_data_size();

    auto input_record_data = device::push_bytes(buffer, data_size);
    if(!input_record_data)
    {
        return false;
    }

    list.capacity = INPUT::MAX_RECORDS;
    list.size = 0;
    list.read_index = 0;
    list.data = (InputRecord*)input_record_data;

    return true;
}