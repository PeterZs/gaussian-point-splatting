#pragma once

#define EPSILON 1e-10

#ifndef SIGTRAP
#define SIGTRAP 5
#endif

#include <thrust/device_vector.h>
#include <csignal>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <string>

// --------------------------------------
// CUDA error checking (debug-only checks)
// --------------------------------------
#ifndef NDEBUG
#define ERRCHECK(ans) do { \
        cudaError_t code_ = (ans); \
        if (code_ != cudaSuccess) { \
            fprintf(stderr, "%s:%d: CUDA API error: %s\n", __FILE__, __LINE__, cudaGetErrorString(code_)); \
            raise(SIGTRAP); \
        } \
    } while(0)

#define LASTERR do { \
        cudaError_t code_ = cudaGetLastError(); \
        if (code_ != cudaSuccess) { \
            fprintf(stderr, "%s:%d: CUDA kernel launch error: %s\n", __FILE__, __LINE__, cudaGetErrorString(code_)); \
            raise(SIGTRAP); \
        } \
    } while(0)
#else
    // In release: always execute the call, just don’t check the result
#define ERRCHECK(ans) (void)(ans)
//#define LASTERR do { /* no-op in release */ } while(0)
#define LASTERR ((void)0)
#endif

// CUDA kernel launch macro with NVCC guard
#if defined(__CUDACC__)
// NVCC sees this: real CUDA launch + debug-only LASTERR()
#  define KERNEL_LAUNCH(kernel, grid, block, ...) \
     do { \
         kernel<<<grid, block>>>(__VA_ARGS__); \
         LASTERR; \
     } while(0)
#else
// MSVC/IntelliSense sees this: dummy call so it parses, but doesn’t try <<< >>>
// Never used for actual compilation when NVCC compiles .cu units.
#  define KERNEL_LAUNCH(kernel, grid, block, ...) \
     do { fprintf(stderr, "ERROR: KERNEL_LAUNCH used in non-CUDA TU: %s\n", __FILE__); raise(SIGTRAP); } while(0)
#endif


// --------------------------------------
// Grid helpers
// --------------------------------------
__host__ __forceinline__
dim3 makeGrid1D(int n, int blockSize) {
    return dim3((n + blockSize - 1) / blockSize);
}

__host__ __forceinline__
dim3 makeGrid2D(int w, int h, int bx, int by) {
    return dim3((w + bx - 1) / bx, (h + by - 1) / by);
}

// --------------------------------------
// Occupancy-based block size suggestion
// --------------------------------------
template <typename KernelT>
inline int suggestBlockSize(KernelT kernel, int dynamicSmemBytes = 0) {
    int minGridSize = 0, blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, dynamicSmemBytes, 0);
    return blockSize > 0 ? blockSize : 256;
}

// --------------------------------------
// Timing helpers (always execute API calls)
// --------------------------------------
inline void preRecord(cudaEvent_t start) {
    ERRCHECK(cudaEventRecord(start, 0));
}

inline void postRecordAndSync(cudaEvent_t start, cudaEvent_t stop, float& outMs) {
    ERRCHECK(cudaEventRecord(stop, 0));
    ERRCHECK(cudaEventSynchronize(stop));
    ERRCHECK(cudaEventElapsedTime(&outMs, start, stop));
}

// --------------------------------------
// Misc
// --------------------------------------
#define ASSIGN_128(dest, src) do { \
    assert(sizeof(src) == sizeof(int4) && sizeof(dest) == sizeof(int4)); \
    *reinterpret_cast<int4*>(&(dest)) = *reinterpret_cast<const int4*>(&(src)); \
} while(0)

__host__ __device__ __forceinline__
unsigned long upperPowerOf2(unsigned long v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

inline std::ostream& operator<<(std::ostream& os, const glm::vec2& vec) {
    os << "{" << vec.x << ", " << vec.y << "}";
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const glm::vec3& vec) {
    os << "{" << vec.x << ", " << vec.y << ", " << vec.z << "}";
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const glm::vec4& vec) {
    os << "{" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << "}";
    return os;
}


