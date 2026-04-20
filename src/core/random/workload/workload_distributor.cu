#include "workload_distributor.cuh"
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/partition.h>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include "utils/utils.cuh"

WorkloadDistributor::WorkloadDistributor(uint32_t max_num_samples, uint32_t max_num_weights) : max_num_samples(max_num_samples), max_num_weights(max_num_weights)
{
	allocate_memory();
}


WorkloadDistributor::~WorkloadDistributor()
{

}

GpuArray<uint32_t>& WorkloadDistributor::getWeights() { return weights; }

GpuArray<uint32_t>& WorkloadDistributor::getIndices()
{
	return indices;
}

GpuArray<WEIGHT_MASK_TYPE>& WorkloadDistributor::getWeightsMaskCurrent()
{
	return iteration % 2 == 0 ? weights_maskA : weights_maskB;
}

GpuArray<WEIGHT_MASK_TYPE>& WorkloadDistributor::getWeightsMaskPrevious()
{
	return iteration % 2 == 0 ? weights_maskB : weights_maskA;
}

uint32_t WorkloadDistributor::getNumSamples() const
{
	return num_samples;
}

uint32_t WorkloadDistributor::getNumWeights() const
{
	return max_num_weights;
}

__global__ void scatter_indices(const uint32_t* mapping, const WEIGHT_MASK_TYPE* mask, uint32_t* indices, int num_weights, int num_points) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_weights && WorkloadDistributor::is_valid_weight(idx, mask)) {
		uint32_t pidx = mapping[idx];
		if (pidx < num_points)
			indices[pidx] = idx;
		//indices[idx]++;
	}
}

void WorkloadDistributor::build() {
	if (max_num_weights == 0) {
		num_samples = 0;
		return;
	}

	//get count of the last element; needed because this won't be included in the prefix sum.
	uint32_t last_count = 0;
	ERRCHECK(cudaMemcpy(&last_count,
		weights.device_data() + (max_num_weights - 1),
		sizeof(uint32_t),
		cudaMemcpyDeviceToHost));

	uint32_t last_index = max_num_weights - 1;
	uint32_t word_idx = last_index / FLAGS_PER_MASK;
	uint32_t bit_idx = last_index % FLAGS_PER_MASK;

	WEIGHT_MASK_TYPE mask_word = 0;
	ERRCHECK(cudaMemcpy(&mask_word,
		getWeightsMaskCurrent().device_data() + word_idx,
		sizeof(WEIGHT_MASK_TYPE),
		cudaMemcpyDeviceToHost));

	if (((mask_word >> bit_idx) & 1u) == 0) {
		last_count = 0;
	}

	uint32_t* wptr = weights.device_data();
	WEIGHT_MASK_TYPE* wmptr = getWeightsMaskCurrent().device_data();

	auto masked_weight = [=] __device__(uint32_t idx) -> uint32_t {
		return is_valid_weight(idx, wmptr) ? wptr[idx] : 0u;
	};
	auto first = thrust::make_transform_iterator(thrust::make_counting_iterator<uint32_t>(0), masked_weight);
	auto last = first + max_num_weights;

	//exclusive prefix sum to create CDF of points/gaussian
	thrust::exclusive_scan(thrust::device,
		first, last,
		weights.device_data(), 0u);

	uint32_t last_offset = 0;
	ERRCHECK(cudaMemcpy(&last_offset,
		weights.device_data() + (max_num_weights - 1),
		sizeof(uint32_t),
		cudaMemcpyDeviceToHost));

	num_samples = glm::clamp(last_offset + last_count, 0u, max_num_samples);

	if (num_samples <= 0) {
		//num_weights = 0;
		num_samples = 0;
		return;
	}

	//map 1 into each index in the above array
	//first clear
	ERRCHECK(cudaMemset(indices.device_data(), 0, num_samples * sizeof(uint32_t)));

	{ // scatter flags
		int blockSize = suggestBlockSize(scatter_indices);
		dim3 grid = makeGrid1D(max_num_weights, blockSize);
		KERNEL_LAUNCH(scatter_indices, grid, blockSize,
			weights.device_data(),
			getWeightsMaskCurrent().device_data(),
			indices.device_data(),
			max_num_weights, num_samples);
	}

	//use a max scan to fill in the gaps, this works assuming that scattered indices are sorted ascendingly
	thrust::inclusive_scan(
		thrust::device,
		indices.device_data(),
		indices.device_data() + num_samples,
		indices.device_data(),
		thrust::maximum<uint32_t>{}
	);

	//reset previous weights mask. this will be used next. NOTE: this is currently handled by the mask
	//auto& prevWeightsMask = getWeightsMaskPrevious();
	//ERRCHECK(cudaMemset(prevWeightsMask.device_data(), 0, prevWeightsMask.size * prevWeightsMask.size()));
	//ERRCHECK(cudaMemset(weights.device_data(), 0, weights.size * weights.size()));
	iteration++;
}

void WorkloadDistributor::allocate_memory()
{
	std::cout << "max_num_weights: " << max_num_weights << ", max_num_samples: " << max_num_samples << std::endl;

	weights.resize(max_num_weights);

	uint32_t num_mask_words = (max_num_weights + 31) / 32;
	weights_maskA.resize(num_mask_words);
	weights_maskB.resize(num_mask_words);

	indices.resize(max_num_samples);
}
