#include "input_record.hpp"


DeviceInputList* make_device_input_list(device::MemoryBuffer& buffer)
{
    auto data_size = device_input_list_data_size();
    auto struct_size = sizeof(DeviceInputList);
    
    auto input_record_data = device::push_bytes(buffer, data_size);
    if(!input_record_data)
    {
        return nullptr;
    }

    auto struct_data = device::push_bytes(buffer, struct_size);
    if(!struct_data)
    {
        device::pop_bytes(buffer, data_size);
        return nullptr;
    }

    DeviceInputList list;
    list.capacity = MAX_INPUT_RECORDS;
    list.size = 0;
    list.read_index = 0;
    list.data = (InputRecord*)input_record_data;    

    auto device_dst = (DeviceInputList*)struct_data;

    if(!cuda_memcpy_to_device(&list, device_dst, struct_size))
    {
        return nullptr;
    }

    return device_dst;
}