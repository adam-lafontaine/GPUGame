#pragma once

#include "../../device/device.hpp"

constexpr u32 MAX_INPUT_RECORDS = 5000;

using uInput = u8;

constexpr uInput INPUT_PLAYER_UP    = 0b00000001;
constexpr uInput INPUT_PLAYER_DOWN  = 0b00000010;
constexpr uInput INPUT_PLAYER_LEFT  = 0b00000100;
constexpr uInput INPUT_PLAYER_RIGHT = 0b00001000;


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
    return sizeof(InputRecord) * MAX_INPUT_RECORDS;
}


inline bool make_device_input_list(DeviceInputList& list, device::MemoryBuffer& buffer)
{
    auto data_size = device_input_list_data_size();

    auto input_record_data = device::push_bytes(buffer, data_size);
    if(!input_record_data)
    {
        assert("make_device_input_list" && false);
        return false;
    }

    list.capacity = MAX_INPUT_RECORDS;
    list.size = 0;
    list.read_index = 0;
    list.data = (InputRecord*)input_record_data;

    return true;
}