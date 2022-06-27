#pragma once

//#define CUDA_NOT_INSTALLED

#define CUDA_PRINT_ERROR

#ifndef CUDA_NOT_INSTALLED

#include <cuda_runtime.h>

#define GPU_KERNAL __global__
#define GPU_FUNCTION __device__
#define HOST_FUNCTION __host__
#define GPU_GLOBAL_VARIABLE __device__
#define GPU_GLOBAL_CONSTANT __constant__
#define GPU_BLOCK_VARIABLE __shared__
#define GPU_UNIFIED __device__ __managed__
#define GPU_CONSTEXPR_FUNCTION __device__ constexpr

#define cuda_barrier __syncthreads

// TODO: streams

#define cuda_launch_kernel(kernel_name, n_blocks, blocksize, ...) \
    kernel_name <<< (n_blocks), (blocksize) >>> (__VA_ARGS__);

#else

#define GPU_KERNAL /*__global__*/
#define GPU_FUNCTION /*__device__*/
#define HOST_FUNCTION /*__host__*/
#define GPU_GLOBAL_VARIABLE /*__device__*/
#define GPU_GLOBAL_CONSTANT /*__constant__*/
#define GPU_BLOCK_VARIABLE /*__shared__*/
#define GPU_UNIFIED /*__device__ __managed__*/
#define GPU_CONSTEXPR_FUNCTION /*__device__ constexpr*/

inline void cuda_barrier(){}

using cudaError_t = int;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 0


inline const char* cudaGetErrorString(cudaError_t){ return "xxx\0"; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, int){ return cudaSuccess; }
inline cudaError_t cudaGetLastError(){ return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize(){ return cudaSuccess; }
inline cudaError_t cudaMalloc(void**, size_t){ return cudaSuccess; }
inline cudaError_t cudaMallocManaged(void**, size_t){ return cudaSuccess; }
inline cudaError_t cudaFree(void *){ return cudaSuccess; }

class FakeCudaPt
{
public:
    int x = 0;
    int y = 0;
    int z = 0;
};

constexpr FakeCudaPt blockDim = { 0, 0, 0 };
constexpr FakeCudaPt blockIdx = { 0, 0, 0 };
constexpr FakeCudaPt threadIdx = { 0, 0, 0 };

// TODO: streams

#define cuda_launch_kernel(kernel_name, n_blocks, blocksize, ...) \
    kernel_name (__VA_ARGS__);

#endif



