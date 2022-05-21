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


constexpr size_t device_input_list_total_size()
{
    return 
        sizeof(DeviceInputList)
        + device_input_list_data_size();
}


DeviceInputList* make_device_input_list(device::MemoryBuffer& buffer);