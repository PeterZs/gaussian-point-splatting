#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#define CUDA_CHECK(x) \
    if ((x) != cudaSuccess) \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(x)))

template <typename T>
class GpuArray
{
public:
    GpuArray() : d_ptr(nullptr), h_ptr(nullptr), _size(0), dirty(false) {}

    explicit GpuArray(size_t count) : d_ptr(nullptr), h_ptr(nullptr), _size(0), dirty(false)
    {
        resize(count);
    }

    ~GpuArray()
    {
        free();
    }

    // Prevent accidental copies (which cause double-frees)
    GpuArray(const GpuArray&) = delete;
    GpuArray& operator=(const GpuArray&) = delete;

    // Support moving resources
    GpuArray(GpuArray&& other) noexcept
        : h_ptr(other.h_ptr), d_ptr(other.d_ptr), _size(other._size), dirty(other.dirty)
    {
        other.h_ptr = nullptr;
        other.d_ptr = nullptr;
        other._size = 0;
    }

    GpuArray& operator=(GpuArray&& other) noexcept
    {
        if (this != &other)
        {
            free();
            h_ptr = other.h_ptr;
            d_ptr = other.d_ptr;
            _size = other._size;
            dirty = other.dirty;
            other.h_ptr = nullptr;
            other.d_ptr = nullptr;
            other._size = 0;
        }
        return *this;
    }

    void resize(size_t count)
    {
        if (_size == count) return;

        free();
        _size = count;

        if (_size > 0) {
            h_ptr = static_cast<T*>(std::malloc(_size * sizeof(T)));
            CUDA_CHECK(cudaMalloc((void**) &d_ptr, _size * sizeof(T)));

            // Initialize memory to match std::vector behavior
            std::memset(h_ptr, 0, _size * sizeof(T));
        }
        dirty = true;
    }

    void free()
    {
        if (h_ptr) std::free(h_ptr);
        if (d_ptr) cudaFree(d_ptr);
        h_ptr = nullptr;
        d_ptr = nullptr;
        _size = 0;
    }

    size_t size() const { return _size; }

    // Host access
    __host__ T& operator[](size_t i)
    {
        dirty = true;
        return h_ptr[i];
    }

    __host__ const T& operator[](size_t i) const
    {
        return h_ptr[i];
    }

    // Device access (Note: No __device__ operator[] because h_ptr is invalid on GPU)
    __host__ T* device_data()
    {
        sync_to_device_if_dirty();
        return d_ptr;
    }

    __host__ const T* device_data() const
    {
        const_cast<GpuArray*>(this)->sync_to_device_if_dirty();
        return d_ptr;
    }

    __host__ T* host_data()
    {
        dirty = true;
        return h_ptr;
    }

    void upload()
    {
        if (_size > 0) {
            CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, _size * sizeof(T), cudaMemcpyHostToDevice));
        }
        dirty = false;
    }

    void download()
    {
        if (_size > 0) {
            CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, _size * sizeof(T), cudaMemcpyDeviceToHost));
        }
        dirty = false;
    }

    void mark_dirty() { dirty = true; }

    void sync_to_device_if_dirty()
    {
        if (dirty) upload();
    }

private:
    T* h_ptr;
    T* d_ptr;
    size_t _size;
    bool dirty;
};