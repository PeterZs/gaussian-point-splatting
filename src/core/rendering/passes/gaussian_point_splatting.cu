#include "gaussian_point_splatting.cuh"
#include "core/random/random.cuh"
#include "core/random/poisson.cuh"
#include "utils/utils.cuh"
#include <random>
#include <timings/stochastic_timings.cuh>
#include <cstdint>
#include <limits>
#include <tuple>
#include <stdexcept>


// --------------------------------------
// Preprocess kernel and wrapper
// --------------------------------------

__global__ void preprocessGaussiansKernel(
	GPUGaussianScene scene,
	const bool onlyPreviouslyOccludedGaussians,
	//Point Weights
	uint32_t* __restrict__ d_gaussianPointWeights,
	WEIGHT_MASK_TYPE* __restrict__ d_gaussianPointWeightMasksCurrent,
	const WEIGHT_MASK_TYPE* __restrict__ d_gaussianPointWeightMasksPrevious,
	//#ifdef ENABLE_FREEZING_CULLING
	WEIGHT_MASK_TYPE* __restrict__ d_gaussianWeightFrameMask,
	const bool freeze_culling,
	//#endif    
	const DepthMipChainDevice depthMipChain,
	const GaussianBVH::GaussianBVHDevice gaussianBVHDev,
	const glm::mat4 vpMatrix,
	const glm::mat4 vMatrix,
	const glm::mat4 camMatrix,
	const glm::vec4* d_frustumPlanes,
	const glm::vec2 tan_fov,
	const glm::vec2 focal,
	const int numGaussians,
	const int up_width,
	const int up_height,
	const int frame_seed,
	const bool enable_occlusion_culling,
	const bool enable_hierarchical_culling,
	const bool cull_small_gaussians,
	const bool use_unbiased_2d_splatting,
	const bool reduce_point_count,
	const int supersampling_factor
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint2 seed = Random::makeSeed(idx, frame_seed);

	bool splats_points = false;
	bool splats_tiles = false;
	do {
		if (idx >= numGaussians) break;

		if (enable_hierarchical_culling && gaussianBVHDev.is_culled(idx)) {
			break;
		}

#ifdef ENABLE_FREEZING_CULLING
		if (freeze_culling && d_gaussianWeightFrameMask[idx] <= 0) break;
#endif  

		// in the second render pass, ignore gaussians that were previously marked as having non-zero importance
		if (onlyPreviouslyOccludedGaussians
			&& (
				WorkloadDistributor::is_valid_weight(idx, d_gaussianPointWeightMasksPrevious)
				)
			) {
			break;
		}

		float opacity = scene.get_opacity(idx);
		if (opacity <= 0.0f) break;

		GaussianGeometry gaussianGeometry = scene.get_gaussian_transform(idx);

		glm::vec3 view_mean = CudaMath::transformPoint4x3(gaussianGeometry.mean, vMatrix);
		if (-view_mean.z <= 0.2f) { // check view space
			break;
		}
		glm::mat3 cov = CudaMath::scale_rot_to_cov(gaussianGeometry.scale, gaussianGeometry.quat);
		glm::vec3 cameraPosition = glm::vec3(camMatrix[3]);

		glm::vec3 cov2d = CudaMath::computeCov2D(vMatrix, cov, gaussianGeometry.mean, tan_fov, focal);
		cov2d += glm::vec3{ 0.3f * supersampling_factor * supersampling_factor, 0.0f, 0.3f * supersampling_factor * supersampling_factor };
		const float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

		if (det <= 0.0f) {
			break;
		}

		glm::vec4 proj_mean = vpMatrix * glm::vec4(gaussianGeometry.mean, 1.0f);
		proj_mean *= 1.0f / proj_mean.w;

		glm::vec2 pixel_mean = (glm::vec2(proj_mean.x, proj_mean.y) * 0.5f + glm::vec2(0.5f))
			* glm::vec2(up_width, up_height);

		glm::ivec2 p_min_bounds{};
		glm::ivec2 p_max_bounds{};
		if (!CudaMath::compute2DGaussianBounds3DGS(p_min_bounds, p_max_bounds, { up_width, up_height }, cov2d, pixel_mean))
			break;

		// Frustum culling
		if (p_min_bounds.x >= up_width || p_min_bounds.y >= up_height || p_max_bounds.x < 0 || p_max_bounds.y < 0) {
			break;
		}

		if (cull_small_gaussians)
		{
			float _trace = cov2d.x + cov2d.z;
			float _traceOver2 = 0.5f * _trace;
			float _term2 = __fsqrt_rn(fmaxf(0.1f, _traceOver2 * _traceOver2 - det));
			float _eigenValue1 = _traceOver2 + _term2;
			float _eigenValue2 = _traceOver2 - _term2;

			if (_eigenValue1 < 0.02f || _eigenValue2 < 0.02f) break;
		}

		constexpr float TWO_PI = 2.0f * 3.14159265358979323846f;

		float importance = 0.0f;
		float s_det = __fsqrt_rn(det);
		if (use_unbiased_2d_splatting) {
			importance = TWO_PI * s_det * Random::dilog(opacity);
		}
		else {
			importance = opacity * TWO_PI * s_det;
		}

		if (enable_occlusion_culling && depth_mip_is_occluded(depthMipChain, proj_mean.z, p_min_bounds / supersampling_factor, p_max_bounds / supersampling_factor)) {
			break;
		}

		float K = static_cast<float>(supersampling_factor * supersampling_factor * POINTS_PER_KERNEL);
		if (!reduce_point_count) K = 1.0f;

		uint32_t expected_num_points = static_cast<uint32_t>(importance);
		uint32_t num_points = 0;
		uint32_t num_tiles = 0;

		if (use_unbiased_2d_splatting) {
			num_points = Random::poisson(seed, importance);
			if (K > 1)
				num_points = Random::stochastic_round(static_cast<float>(num_points) / K, Random::pcg2d(seed).x);
		}
		else {
			num_points = Random::stochastic_round(importance / K, Random::pcg2d(seed).x);
		}

		if (num_points > 0) {
			num_points = glm::min(num_points, (uint32_t)(up_width * up_height / (2 * K)));
			d_gaussianPointWeights[idx] = num_points;
			splats_points = true;
		}
		else
			break;

#ifdef ENABLE_FREEZING_CULLING
		d_gaussianWeightFrameMask[idx] = 1;
#endif 

#ifdef DISABLE_SPHERICAL_HARMONICS
		break;
#endif // DISABLE_SPHERICAL_HARMONICS

		scene.set_color(idx, scene.get_sh(idx, 4, gaussianGeometry.mean, cameraPosition));
	} while (false);

	//__syncwarp();

	uint32_t points_mask = __ballot_sync(0xffffffff, splats_points);
	if ((threadIdx.x & 31) == 0) {
		uint32_t warpBase = idx & ~31u;
		uint32_t wordIdx = warpBase >> 5;
		if (warpBase < numGaussians) {
			d_gaussianPointWeightMasksCurrent[wordIdx] = points_mask;
		}
	}
}

static void launch_preprocessKernel(GPUGaussianScene& scene, const bool onlyPreviouslyOccludedGaussians,
	//Point Weights
	uint32_t* __restrict__ d_gaussianPointWeights,
	WEIGHT_MASK_TYPE* __restrict__ d_gaussianPointWeightMasksCurrent,
	const WEIGHT_MASK_TYPE* __restrict__ d_gaussianPointWeightMasksPrevious,
#ifdef ENABLE_FREEZING_CULLING
	WEIGHT_MASK_TYPE* __restrict__ d_gaussianWeightFrameMask,
	bool freeze_culling,
#endif      
	const DepthMipChain& depthMipChain,
	const GaussianBVH& gaussianBVH,
	const glm::mat4& vpMatrix,
	const glm::mat4& vMatrix,
	const glm::mat4& camMatrix,
	const glm::vec4* d_frustumPlanes,
	const glm::vec2& tan_fov,
	const glm::vec2& focal,
	const int numGaussians,
	const int up_width, const int up_height, const int frame_seed,
	const bool enable_occlusion_culling,
	const bool enable_hierarchical_culling,
	const bool cull_small_gaussians,
	const bool use_unbiased_2d_splatting,
	const bool reduce_point_count,
	const int supersampling_factor)
{

#ifndef ENABLE_FREEZING_CULLING
	WEIGHT_MASK_TYPE* d_gaussianWeightFrameMask = nullptr;
	bool freeze_culling = false;
#endif      

	int blockSize = 256;
	dim3 grid = makeGrid1D(numGaussians, blockSize);

	KERNEL_LAUNCH(preprocessGaussiansKernel, grid, blockSize,
		scene,
		onlyPreviouslyOccludedGaussians,
		d_gaussianPointWeights,
		d_gaussianPointWeightMasksCurrent,
		d_gaussianPointWeightMasksPrevious,
		d_gaussianWeightFrameMask,
		freeze_culling,
		depthMipChain.device_view(),
		gaussianBVH.device_view(),
		vpMatrix,
		vMatrix,
		camMatrix,
		d_frustumPlanes,
		tan_fov,
		focal,
		numGaussians,
		up_width,
		up_height,
		frame_seed,
		enable_occlusion_culling,
		enable_hierarchical_culling,
		cull_small_gaussians,
		use_unbiased_2d_splatting,
		reduce_point_count,
		supersampling_factor
	);
}

// --------------------------------------
// Points Splat kernel and wrapper (2D/3D)
// --------------------------------------

__global__ void splatGaussianPointsKernel(
	const GPUGaussianScene scene,
	const uint32_t* __restrict__ d_gaussian_indices,
	uint64_t* __restrict__ depthBuffer,
	const glm::mat4 vpMatrix,
	const glm::mat4 vMatrix,
	const glm::mat4 camMatrix,
	const glm::vec2 tan_fov,
	const glm::vec2 focal,
	const int up_width,
	const int up_height,
	const int numPasses,
	const int frame_seed,
	const int numPoints,
	const float near_plane,
	const float far_plane,
	const bool use_unbiased_2d_splatting,
	const bool reduce_point_count,
	const int supersampling_factor
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPoints) return;

	uint2 seed = Random::makeSeed(idx, frame_seed);
	uint32_t g_i = d_gaussian_indices[idx];

	// Gaussian transform and color
	GaussianGeometry gaussianGeometry = scene.get_gaussian_transform(g_i);
	uint64_t color = scene.get_color(g_i);

	glm::vec4 proj_mean = vpMatrix * glm::vec4(gaussianGeometry.mean, 1.0f);
	proj_mean /= proj_mean.w;
	if (proj_mean.z < -1.0f || proj_mean.z > 1.0f) return;

	// Shared covariance in object space
	glm::mat3 cov = CudaMath::scale_rot_to_cov(gaussianGeometry.scale, gaussianGeometry.quat);
	// Screen-space Gaussian parameters
	glm::vec3 cov2d = CudaMath::computeCov2D(vMatrix, cov, gaussianGeometry.mean, tan_fov, focal);
	cov2d += glm::vec3{ 0.3f * supersampling_factor * supersampling_factor, 0.0f, 0.3f * supersampling_factor * supersampling_factor }; // slight blur

	const float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
	if (det <= 0) return;

	glm::vec3 conic = CudaMath::conic(glm::vec3{ cov2d.x, cov2d.y, cov2d.z }, 1.0f / det);

	const float c0 = __fsqrt_rn(cov2d.x);
	const float c1 = cov2d.y / c0;
	const float c2 = __fsqrt_rn(cov2d.z - c1 * c1);

	const glm::vec2 pixel_mean = (glm::vec2(proj_mean.x, proj_mean.y) * 0.5f + glm::vec2(0.5f))
		* glm::vec2(up_width, up_height);

	glm::ivec2 p_min_bounds{};
	glm::ivec2 p_max_bounds{};
	if (!CudaMath::compute2DGaussianBounds3DGS(p_min_bounds, p_max_bounds, { up_width, up_height }, cov2d, pixel_mean))
		return;

	glm::vec3 view_mean = CudaMath::transformPoint4x3(gaussianGeometry.mean, vMatrix);
	const uint64_t view_depth_bits = CudaMath::convert_view_depth(view_mean.z, near_plane, far_plane) << DEPTH_SHIFT;
	float opacity = scene.get_opacity(g_i);

	int K = supersampling_factor * supersampling_factor * POINTS_PER_KERNEL;
	if (!reduce_point_count) K = 1;
#pragma unroll
	for (size_t p = 0; p < numPasses * K; ++p)
	{
		float2 u = Random::pcg2d(seed);
		float2 xy;
		if (use_unbiased_2d_splatting) {
			xy = Random::correctedBoxMuller(u.x, u.y, opacity);
		}
		else {
			xy = Random::boxMuller(u.x, u.y);
		}

		glm::vec2 pixel(pixel_mean.x + c0 * xy.x, pixel_mean.y + c1 * xy.x + c2 * xy.y);
		pixel = glm::floor(pixel);// -glm::vec2{ 0.5f, 0.f };
		int x = static_cast<int>(pixel.x);
		int y = static_cast<int>(pixel.y);

		glm::vec2 delta(pixel_mean - pixel);

		//reject if outside of screen/tile grid aligned bounds of gaussian
		if (x < p_min_bounds.x || x >= p_max_bounds.x || y < p_min_bounds.y || y >= p_max_bounds.y) continue;

		//rejection mahalanobis sampling
		{
			float gaussian_weight = CudaMath::gaussian_value(conic, delta);
			float alpha = opacity * gaussian_weight;
			if (alpha < 1.0f / 255.0f)// || Random::pcg2d(seed).x > 0.99f / alpha)
				continue;
		}
		int index = posToMemory(x, y, p / K, numPasses, up_width, up_height);

		if (depthBuffer[index] <= view_depth_bits) continue;

		if (use_unbiased_2d_splatting) {
			//atomicMin(&depthBuffer[index], view_depth_bits | color);
			atomicMin(reinterpret_cast<unsigned long long*>(&depthBuffer[index]),
				static_cast<unsigned long long>(view_depth_bits | color));
		}
		else {
			atomicAddColor(&depthBuffer[index], view_depth_bits | color, true);
		}
	}
}

static void launch_splatPointsKernel(GPUGaussianScene& scene,
	WorkloadDistributor* pointsWorkload,
	uint64_t* d_imageBuffer,
	const glm::mat4& vpMatrix,
	const glm::mat4& vMatrix,
	const glm::mat4& camMatrix,
	const glm::vec2& tan_fov,
	const glm::vec2& focal,
	const int up_width,
	const int up_height,
	const int numPasses,
	const int frame_seed,
	const int numPoints,
	const float near_plane,
	const float far_plane,
	const bool use_unbiased_2d_splatting,
	const bool reduce_point_count,
	const int supersampling_factor)
{
	const int blockSize = suggestBlockSize(splatGaussianPointsKernel);
	dim3 grid = makeGrid1D(numPoints, blockSize);

	uint32_t* d_gaussianIndices = pointsWorkload->getIndices().device_data();

	KERNEL_LAUNCH(splatGaussianPointsKernel, grid, blockSize,
		scene,
		d_gaussianIndices,
		d_imageBuffer,
		vpMatrix,
		vMatrix,
		camMatrix,
		tan_fov,
		focal,
		up_width,
		up_height,
		numPasses,
		frame_seed,
		numPoints,
		near_plane,
		far_plane,
		use_unbiased_2d_splatting,
		reduce_point_count,
		supersampling_factor
	);
}

// --------------------------------------
// Screenspace passes
// --------------------------------------

template<bool depthOnly, bool keepOldIfInitialValue>
__global__ void processImageBuffer(
	const uint64_t* __restrict__ imageBuffer,
	glm::vec3* __restrict__ colorBuffer,
	float* __restrict__ depthBuffer,
	const int width,
	const int height,
	const int numPasses,
	const int imageBufferSize,
	const float near_plane,
	const float far_plane,
	const int supersampling_factor,
	const uint64_t initial_value
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	uint32_t idx = y * width + x;

	int buffer_x = x * supersampling_factor;
	int buffer_y = y * supersampling_factor;

	int up_width = width * supersampling_factor;
	int up_height = height * supersampling_factor;

	float depth = 1e9f;// depthBuffer[idx];// 1e9f;
	glm::vec3 color = glm::vec3(0.0);
	bool any_points = false;
#pragma unroll
	for (int dx = 0; dx < supersampling_factor; dx++)
#pragma unroll
		for (int dy = 0; dy < supersampling_factor; dy++)
			for (int p = 0; p < numPasses; ++p) {
				int i = posToMemory(buffer_x + dx, buffer_y + dy, p, numPasses, up_width, up_height);
				if (i >= imageBufferSize) continue;

				uint64_t val = imageBuffer[i];
				if (val != initial_value)
					any_points = true;

				float view_depth = CudaMath::convert_view_depth((uint64_t)((val >> 36ull) & DEPTH_MAX), near_plane, far_plane);
				float z_ndc = CudaMath::view_to_projected_depth(view_depth, near_plane, far_plane);
				depth = glm::min(depth, z_ndc);

				if constexpr (depthOnly) continue;

				//36 bits
				color.r += static_cast<float>((val >> 24) & 0xFFF) / 255.0f;
				color.g += static_cast<float>((val >> 12) & 0xFFF) / 255.0f;
				color.b += static_cast<float>((val >> 0) & 0xFFF) / 255.0f;
			}

	// Store averaged results
	float invPasses = 1.0f / (numPasses * supersampling_factor * supersampling_factor);
	if constexpr (!depthOnly) {
		colorBuffer[idx] = color * invPasses; //*10.0f;
	}
	if constexpr (keepOldIfInitialValue) {
		if (any_points)
			depthBuffer[idx] = depth;
	}
	else
		depthBuffer[idx] = depth;
}

template<bool depthOnly, bool keepOldIfInitialValue>
static void launch_combineKernel(uint64_t* d_imageBuffer,
	glm::vec3* d_image,
	float* d_depth,
	int width, int height,
	int numPasses, int imageBufferSize,
	const float near_plane, const float far_plane,
	const int supersampling_factor,
	uint64_t initial_value)
{
	auto kernel = processImageBuffer<depthOnly, keepOldIfInitialValue>;
	//const int blockSize = suggestBlockSize(kernel);
	dim3 blockSize(16, 16);
	dim3 grid = makeGrid2D(width, height, blockSize.x, blockSize.y);

	KERNEL_LAUNCH(kernel, grid, blockSize,
		d_imageBuffer,
		d_image,
		d_depth,
		width,
		height,
		numPasses,
		imageBufferSize,
		near_plane,
		far_plane,
		supersampling_factor,
		initial_value
	);
}

__global__ void clearImageBuffer(uint64_t* __restrict__ buffer, const uint64_t value, const int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		buffer[idx] = value;
	}
}

static void launch_clearImageBuffer(uint64_t* d_buffer,
	uint64_t value,
	int imageBufferSize) {
	int blockSize = suggestBlockSize(clearImageBuffer);
	dim3 grid = makeGrid1D(imageBufferSize, blockSize);
	KERNEL_LAUNCH(clearImageBuffer, grid, blockSize, d_buffer, value, imageBufferSize);
}


// --------------------------------------
// GaussianPointSplatting: render()
// --------------------------------------
void GaussianPointSplatting::render(glm::vec3* d_image,
	float* d_depth,
	const RenderSettings& settings)
{
	update_image_buffers(settings);

	auto vMatrix = settings.cameraViewMatrix;
	auto camMatrix = glm::affineInverse(vMatrix);
	auto vpMatrix = settings.cameraProjectionMatrix * vMatrix;

	float r = settings.get_aspectXY();
	float t_f_y = glm::tan(glm::radians(settings.fovy) * 0.5f);
	float fovx = 2.0f * glm::atan(t_f_y * r);
	float t_f_x = glm::tan(fovx * 0.5f);
	glm::vec2 tan_fov{ t_f_x, t_f_y };
	glm::vec2 focal{ settings.resolution.x * settings.supersampling_factor / (2.0f * tan_fov.x),
					 settings.resolution.y * settings.supersampling_factor / (2.0f * tan_fov.y) };

	const float pixel_count = float(settings.resolution.x * settings.supersampling_factor) * float(settings.resolution.y * settings.supersampling_factor);

	settings.statistics->numPoints = 0;
	uint64_t backGroundColor = 0xFFFFFFF000000000;
	backGroundColor |= CudaMath::vec3To36Bits(settings.backgroundColor);

	uint64_t* current_image_buffer = get_current_image_buffer(settings.frame);
	uint64_t* previous_image_buffer = get_previous_image_buffer(settings.frame);

	launch_clearImageBuffer(current_image_buffer,
		backGroundColor,
		imageBufferSize);

#ifdef ENABLE_FREEZING_CULLING
	//clear pre frame data if not frozen
	if (!settings.freeze_culling) {
		cudaMemset(per_frame_weights_mask.device_data(), 0, gaussians.size() * sizeof(WEIGHT_MASK_TYPE));
	}
#endif    

	// handle preprocessing and splatting
	auto renderPass = [&](TimingSystem::StochasticTiming& timings, const bool isFirstPass) {
		int frame = 4 * settings.frame + (isFirstPass ? 0 : 2);

		auto enable_hierarchical_culling = settings.enable_hierarchical_culling && !settings.fully_disable_hierarchical_culling;

		if (enable_hierarchical_culling)
			gaussianBVH.update_hierarchical_culling(settings.cameraPosition, settings.d_frustumPlanes, vpMatrix, depthMipChain, settings.enable_occlusion_culling);

		// preprocess        
		launch_preprocessKernel(gpuGaussianScene, !isFirstPass,
			pointsWorkload->getWeights().device_data(),
			pointsWorkload->getWeightsMaskCurrent().device_data(),
			pointsWorkload->getWeightsMaskPrevious().device_data(),
#ifdef ENABLE_FREEZING_CULLING
			per_frame_weights_mask.device_data(),
			settings.freeze_culling,
#endif    
			depthMipChain,
			gaussianBVH,
			vpMatrix, vMatrix, camMatrix, settings.d_frustumPlanes,
			tan_fov, focal,
			numGaussians,
			settings.resolution.x * settings.supersampling_factor, settings.resolution.y * settings.supersampling_factor, frame,
			settings.enable_occlusion_culling, enable_hierarchical_culling, settings.cull_small_gaussians,
			settings.use_unbiased_2d_splatting,
			settings.reduce_point_count,
			settings.supersampling_factor);

		pointsWorkload->build();
		uint32_t totalWeight = pointsWorkload->getNumSamples();
		int pointCount = totalWeight;
		int num_gaussians = pointsWorkload->getNumWeights();


		settings.statistics->totalWeight = totalWeight;

		if (isFirstPass) {
			settings.statistics->numPointsOne = pointCount;
			settings.statistics->numGaussiansOne = num_gaussians;
		}
		else {
			settings.statistics->numPointsTwo = pointCount;
			settings.statistics->numGaussiansTwo = num_gaussians;
		}
		timings.PointCount = pointCount;

		if (pointCount > 0) {
			launch_splatPointsKernel
			(gpuGaussianScene,
				pointsWorkload,
				current_image_buffer,
				vpMatrix, vMatrix, camMatrix, tan_fov, focal,
				settings.resolution.x * settings.supersampling_factor, settings.resolution.y * settings.supersampling_factor,
				numPasses,
				frame + 1,
				pointCount,
				settings.near_view_plane,
				settings.far_view_plane,
				settings.use_unbiased_2d_splatting,
				settings.reduce_point_count,
				settings.supersampling_factor);
		}
	};

	// -------- First render pass, use reprojected depth occlusion --------
	renderPass(TimingSystem::StochasticTimings::getCurrentTiming(), true);

#ifndef  NDEBUG
	std::cout << "1st Pass: " << TimingSystem::StochasticTimings::getCurrentTiming().PointCount << std::endl;
#endif // ! NDEBUG        

	if (settings.enable_occlusion_culling) {
		launch_combineKernel<true, false>(current_image_buffer, d_image, d_depth,
			settings.resolution.x, settings.resolution.y, numPasses, imageBufferSize,
			settings.near_view_plane, settings.far_view_plane,
			settings.supersampling_factor,
			backGroundColor);
		//launch_inPaintDepthKernel(d_depth, d_tempDepthBuffer, settings.resolution.x, settings.resolution.y, 3);
		depthMipChain.build_mipchain(d_depth, settings.occlusion_culling_min_reduce_count);

		// -------- Second pass render Gaussians that were occluded previously, but now are not --------
		renderPass(TimingSystem::StochasticTimings::getCurrentTiming(), false);
#ifndef  NDEBUG
		std::cout << "2nd Pass: " << TimingSystem::StochasticTimings::getCurrentTiming().PointCount << std::endl;
#endif // ! NDEBUG    
	}

	launch_combineKernel<false, false>(current_image_buffer, d_image, d_depth,
		settings.resolution.x, settings.resolution.y, numPasses, imageBufferSize, settings.near_view_plane, settings.far_view_plane,
		settings.supersampling_factor,
		backGroundColor);

	if (settings.enable_occlusion_culling) {
		//launch_inPaintDepthKernel(d_depth, d_tempDepthBuffer, settings.resolution.x, settings.resolution.y, 3);
		depthMipChain.build_mipchain(d_depth, settings.occlusion_culling_min_reduce_count);
	}
}

uint64_t* GaussianPointSplatting::get_current_image_buffer(int frame_index) const
{
	return frame_index % 2 == 0 ? d_imageBufferA : d_imageBufferB;
}

uint64_t* GaussianPointSplatting::get_previous_image_buffer(int frame_index) const
{
	return frame_index % 2 == 0 ? d_imageBufferB : d_imageBufferA;
}

void GaussianPointSplatting::update_image_buffers(const RenderSettings& settings) {
#ifdef MORTON_ORDER_IMAGE_BUFFER
	int tiles_per_row = (settings.resolution.x * settings.supersampling_factor + MORTON_ORDER_TILE_SIZE - 1) >> MORTON_ORDER_TILE_BITS;
	int tiles_per_col = (settings.resolution.y * settings.supersampling_factor + MORTON_ORDER_TILE_SIZE - 1) >> MORTON_ORDER_TILE_BITS;

	size_t padded_pixels = static_cast<size_t>(tiles_per_row) *
		static_cast<size_t>(tiles_per_col) *
		MORTON_ORDER_TILE_SIZE * MORTON_ORDER_TILE_SIZE;
	uint32_t wishImageBufferSize = padded_pixels * static_cast<size_t>(numPasses);
#else
	uint32_t wishImageBufferSize = static_cast<size_t>(numPasses) * settings.resolution.x * settings.resolution.y;
#endif // MORTON_ORDER_IMAGE_BUFFER

	if (wishImageBufferSize == imageBufferSize) return;

	imageBufferSize = wishImageBufferSize;

	if (d_imageBufferA) ERRCHECK(cudaFree(d_imageBufferA));
	if (d_imageBufferB) ERRCHECK(cudaFree(d_imageBufferB));

	d_imageBufferA = nullptr;
	d_imageBufferB = nullptr;

	uint32_t N = wishImageBufferSize * sizeof(uint64_t);
	ERRCHECK(cudaMalloc((void**)(&d_imageBufferA), N));
	ERRCHECK(cudaMalloc((void**)(&d_imageBufferB), N));
}

GaussianPointSplatting::GaussianPointSplatting(const RenderSettings& settings)
	: numPasses(settings.render_pass_count), maxPointCount(settings.max_number_of_points), depthMipChain(settings.resolution.x, settings.resolution.y)
{
	gaussians.clear();
	std::vector<PlyComment> comments;
	auto sort_morton_order = settings.sort_morton_order && !settings.fully_disable_hierarchical_culling;
	GaussianLoader::loadPlyFile(settings.model_path, gaussians, comments, sort_morton_order);
	bool is_presorted = PlyCommentUtils::hasBVHSorted(comments);
	std::cout << "is pre-sorted: " << (is_presorted ? "True" : "false") << std::endl;

	numGaussians = static_cast<int>(gaussians.size());
	std::cout << "Loaded Gaussian Count: " << numGaussians << std::endl;

	pointsWorkload = new WorkloadDistributor(maxPointCount, numGaussians);

	if (!settings.fully_disable_hierarchical_culling) {
		if (!settings.sort_morton_order) {
			if (!is_presorted) {
				gaussianBVH.build_bvh_and_reorder(gaussians);
				gaussianBVH.release_bvh_host_data();
				std::cout << "Sorted Gaussians." << std::endl;
			}
		}
		gaussianBVH.build_hierarchical_structure(gaussians);
	}

#ifdef ENABLE_FREEZING_CULLING
	per_frame_weights_mask.resize(gaussians.size());
#endif    

	//std::cout << "Tree Span: " << hierarchicalCulling->get_device_view().tree_span << std::endl;
	gpuGaussianScene.upload_from_cpu(gaussians);

	ERRCHECK(cudaEventCreate(&start));
	ERRCHECK(cudaEventCreate(&stop));

	update_image_buffers(settings);

	ERRCHECK(cudaMalloc((void**)(&d_tempDepthBuffer), settings.resolution.x * settings.resolution.y * sizeof(float)));
}

GaussianPointSplatting::~GaussianPointSplatting() {
	if (d_imageBufferA) ERRCHECK(cudaFree(d_imageBufferA));
	if (d_imageBufferB) ERRCHECK(cudaFree(d_imageBufferB));
	if (d_tempDepthBuffer) ERRCHECK(cudaFree(d_tempDepthBuffer));
	d_imageBufferA = nullptr;
	d_imageBufferB = nullptr;
	d_tempDepthBuffer = nullptr;

	gpuGaussianScene.free();

	ERRCHECK(cudaEventDestroy(start));
	ERRCHECK(cudaEventDestroy(stop));

	if (pointsWorkload) {
		delete pointsWorkload;
		pointsWorkload = nullptr;
	}
}
