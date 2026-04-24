#pragma once

#include "config.h"

#include "core/rendering/render_pass.cuh"
#include <cuda_runtime.h>
#include <cuda.h>

#define GLM_FORCE_INTRINSICS
#define GLM_FORCE_PURE
#define GLM_FORCE_CTOR_INIT
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_EXPLICIT_CTOR
#define GLM_FORCE_FAST_MATH
#define GLM_ENABLE_EXPERIMENTAL

#define GLM_FORCE_CUDA
#define GLM_FORCE_CUDA_DEVICE
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include "core/math/linalg.cuh"
#include <cmath>
#include <glm/ext.hpp>
#include "core/math/sh.cuh"
#include "core/rendering/passes/post_processing_kernels.cuh"
#include <vector>

#include "core/gaussians/gaussian_loader.cuh"
#include "core/gaussians/gaussian_types.cuh"
#include "core/rendering/occlusion/depth_mip_chain.cuh"
#include "core/random/workload/workload_distributor.cuh"
#include "core/rendering/occlusion/gaussian_bvh.cuh"

class GaussianPointSplatting : public RenderPass {
public:
	GaussianPointSplatting(const RenderSettings& settings);
	~GaussianPointSplatting();
	void render(glm::vec3* d_image, float* d_depth, const RenderSettings& settings) override;

private:
	std::vector<CPUGaussian> gaussians = {};

	int maxPointCount;
	int numPasses;
	int numGaussians;

	float* d_tempDepthBuffer = nullptr;

	uint64_t* d_imageBufferA = nullptr;
	uint64_t* d_imageBufferB = nullptr;
	uint32_t imageBufferSize = 0;

	uint64_t* get_current_image_buffer(int frame_index) const;
	uint64_t* get_previous_image_buffer(int frame_index) const;

	GPUGaussianScene gpuGaussianScene;

	void update_image_buffers(const RenderSettings& settings);

#ifdef ENABLE_FREEZING_CULLING
	GpuArray<WEIGHT_MASK_TYPE> per_frame_weights_mask = GpuArray<WEIGHT_MASK_TYPE>();
#endif

	WorkloadDistributor* pointsWorkload = nullptr;

public:
	GaussianBVH gaussianBVH{};
	DepthMipChain depthMipChain;

	//timing
	cudaEvent_t start, stop;
};