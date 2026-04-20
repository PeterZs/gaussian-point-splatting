#pragma once

#include <thrust/device_vector.h>
#include <curand_kernel.h>
#include "config.h"
#include "utils/gpu_array.cuh"

class WorkloadDistributor {
public:
    WorkloadDistributor(uint32_t max_num_samples, uint32_t max_num_weights);
    ~WorkloadDistributor();

    GpuArray<uint32_t>& getWeights();
    GpuArray<uint32_t>& getIndices();
    GpuArray<WEIGHT_MASK_TYPE>& getWeightsMaskCurrent();
    GpuArray<WEIGHT_MASK_TYPE>& getWeightsMaskPrevious();
    uint32_t getNumSamples() const;
    uint32_t getNumWeights() const;
    void build();
    void allocate_memory();

    const static int FLAGS_PER_MASK = sizeof(WEIGHT_MASK_TYPE) * 8;
    __device__ __forceinline__ static bool is_valid_weight(const uint32_t index, const WEIGHT_MASK_TYPE* d_mask) {
        uint32_t word_idx = index / FLAGS_PER_MASK;
        uint32_t flag_idx = index % FLAGS_PER_MASK;
        return ((d_mask[word_idx] >> flag_idx) & 1u) != 0;
    }

private:
    //size G
    GpuArray<uint32_t> weights = GpuArray<uint32_t>();

    //size G / 32
    GpuArray<WEIGHT_MASK_TYPE> weights_maskA = GpuArray<WEIGHT_MASK_TYPE>();
    GpuArray<WEIGHT_MASK_TYPE> weights_maskB = GpuArray<WEIGHT_MASK_TYPE>();

    //size P
    GpuArray<uint32_t> indices = GpuArray<uint32_t>();

    uint32_t max_num_samples = 0;
    uint32_t num_samples = 0;

    uint32_t max_num_weights = 0;
    //uint32_t num_weights = 0;

    uint32_t iteration = 0;
    WEIGHT_TYPE total_weight = 0;
};

