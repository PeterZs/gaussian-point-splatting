#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <cstdint>
#include "core/math/sh.cuh"
#include "core/math/linalg.cuh"
#include "core/gaussians/gaussian_types.cuh"
#include "config.h"

__device__ inline bool atomicAddColor(uint64_t* address, uint64_t value, bool add_if_equal) {
	const uint64_t target_depth = value & DEPTH_MASK;
	uint64_t old = *address, assumed;	

	do {
		assumed = old;

		uint64_t assumed_depth = (assumed & DEPTH_MASK);
		uint64_t new_value = value;

		if (target_depth > assumed_depth) {
			return true; //Is occluded, so ignore.
		}
		else if (target_depth == assumed_depth) { // Add color
			if (!add_if_equal) return false;
			uint64_t new_lower = (assumed & COLOR_MASK) + (value & COLOR_MASK);
			new_value = (assumed & DEPTH_MASK) | (new_lower & COLOR_MASK);
		}
		else if (target_depth < assumed_depth) { //Occlude previous
			new_value = value;
		}
		//old = atomicCAS(address, assumed, new_value);
		// Explicitly cast to the signature CUDA expects
		old = atomicCAS(reinterpret_cast<unsigned long long*>(address),
			static_cast<unsigned long long>(assumed),
			static_cast<unsigned long long>(new_value));
	} while (assumed != old);

	return true;
}

//it does keep track of the minimum depth, but adds to the color part
__device__ inline bool atomicAddColorOverDraw(uint64_t* address, uint64_t value) {
	//return atomicMin(address, value);
	const uint64_t target_depth = value & DEPTH_MASK;
	const uint64_t target_color = value & COLOR_MASK;
	uint64_t old = *address, assumed;

	do {
		assumed = old;

		uint64_t assumed_depth = assumed & DEPTH_MASK;
		uint64_t assumed_color = assumed & COLOR_MASK;

		uint64_t new_depth = ((target_depth < assumed_depth) ? target_depth : assumed_depth) & DEPTH_MASK;
		uint64_t new_color = (assumed_color + target_color) & COLOR_MASK;

		uint64_t new_value = new_depth | new_color;

		//old = atomicCAS(address, assumed, new_value);
		old = atomicCAS(reinterpret_cast<unsigned long long*>(address),
			static_cast<unsigned long long>(assumed),
			static_cast<unsigned long long>(new_value));
	} while (assumed != old);

	return true;
}

__device__ __host__ inline uint32_t mortonEncode(const uint32_t x, const uint32_t y) {
	auto spread = [](uint32_t a) {
		a = (a | (a << 8)) & 0x00FF00FF;
		a = (a | (a << 4)) & 0x0F0F0F0F;
		a = (a | (a << 2)) & 0x33333333;
		a = (a | (a << 1)) & 0x55555555;
		return a;
		};
	return spread(x) | (spread(y) << 1);
}

__device__ inline int posToMemory(const int x, const int y, const int p,
	const int numPasses, const int width, const int height) {

	//N = width * height * numPasses;
#ifndef MORTON_ORDER_IMAGE_BUFFER
	return (p * height + y) * width + x;
	//return (y * width + x) * numPasses + p;
#endif
	const int tx = x >> MORTON_ORDER_TILE_BITS;
	const int ty = y >> MORTON_ORDER_TILE_BITS;

	const int lx = x & (MORTON_ORDER_TILE_SIZE - 1);
	const int ly = y & (MORTON_ORDER_TILE_SIZE - 1);

	const uint32_t pixel_code = mortonEncode(lx, ly);

	const int tiles_per_row = (width + MORTON_ORDER_TILE_SIZE - 1) >> MORTON_ORDER_TILE_BITS;
	const int tiles_per_col = (height + MORTON_ORDER_TILE_SIZE - 1) >> MORTON_ORDER_TILE_BITS;

	const int tile_index = ty * tiles_per_row + tx;

	return (tile_index * MORTON_ORDER_TILE_SIZE * MORTON_ORDER_TILE_SIZE + pixel_code) * numPasses + p;
}


// Progressive rendering kernel
__global__ inline void progressiveRender(
	unsigned char* __restrict__ image,
	glm::vec3* __restrict__ currentColor,
	glm::vec3* __restrict__ accumBuffer,
	int width,
	int height,
	int numProgressiveFrames
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= width * height) return;

	glm::vec3 color = currentColor[idx];

	// Accumulate if needed
	if (numProgressiveFrames > 1) {
		color += accumBuffer[idx];
	}

	// Update accumulation buffer
	currentColor[idx] = color;
	accumBuffer[idx] = glm::vec3(0.0f, 0.0f, 0.0f);

	// Final output
	color *= 255.0f / static_cast<float>(numProgressiveFrames);
	color = glm::clamp(color, 0.0f, 255.0f);
	color = glm::round(color);

	
	int base_idx = 3 * idx;
	image[base_idx] = static_cast<uint8_t>(color.r);
	image[base_idx + 1] = static_cast<uint8_t>(color.g);
	image[base_idx + 2] = static_cast<uint8_t>(color.b);
}


__device__ inline glm::vec4 bilinearSampleCombined(
	const glm::vec3* colorBuffer,  // Color history buffer (3 floats per pixel)
	const float* depthBuffer,  // Depth history buffer (1 float per pixel)
	int width,
	int height,
	float px,
	float py
) {
	int x0 = static_cast<int>(floorf(px));
	int y0 = static_cast<int>(floorf(py));
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	x0 = (x0 < 0) ? 0 : ((x0 >= width) ? width - 1 : x0);
	y0 = (y0 < 0) ? 0 : ((y0 >= height) ? height - 1 : y0);
	x1 = (x1 < 0) ? 0 : ((x1 >= width) ? width - 1 : x1);
	y1 = (y1 < 0) ? 0 : ((y1 >= height) ? height - 1 : y1);

	float tx = px - floorf(px);
	float ty = py - floorf(py);

	int idx00 = (y0 * width + x0);
	int idx10 = (y0 * width + x1);
	int idx01 = (y1 * width + x0);
	int idx11 = (y1 * width + x1);

	float d00 = depthBuffer[idx00];
	float d10 = depthBuffer[idx10];
	float d01 = depthBuffer[idx01];
	float d11 = depthBuffer[idx11];

	float topDepth = glm::mix(d00, d10, tx);
	float bottomDepth = glm::mix(d01, d11, tx);
	float depth = glm::mix(topDepth, bottomDepth, ty);

	glm::vec3 c00 = colorBuffer[idx00];
	glm::vec3 c10 = colorBuffer[idx10];
	glm::vec3 c01 = colorBuffer[idx01];
	glm::vec3 c11 = colorBuffer[idx11];

	glm::vec3 top = glm::mix(c00, c10, tx);
	glm::vec3 bottom = glm::mix(c01, c11, tx);
	glm::vec3 color = glm::mix(top, bottom, ty);

	// Return the combined result: RGB from color, and depth in W
	return glm::vec4(color, depth);
}

__device__ __forceinline__
void computeNeighborhoodMinMax(
	const glm::vec3* __restrict__ colorBuffer,
	int x, int y,
	int width, int height,
	glm::vec3& outMin,
	glm::vec3& outMax)
{
	outMin = glm::vec3(1e9f);
	outMax = glm::vec3(-1e9f);

#pragma unroll
	for (int dy = -1; dy <= 1; dy++) {
		int yy = y + dy;
		if (yy < 0 || yy >= height) continue;

#pragma unroll
		for (int dx = -1; dx <= 1; dx++) {
			int xx = x + dx;
			if (xx < 0 || xx >= width) continue;

			glm::vec3 c = colorBuffer[yy * width + xx];

			outMin = glm::min(outMin, c);
			outMax = glm::max(outMax, c);
		}
	}
}


// Temporal reprojection kernel
__global__ inline void temporalRender(
	unsigned char* __restrict__ image,
	glm::vec3* __restrict__ currentColor,
	const glm::vec3* __restrict__ prevColor,
	float* __restrict__ currentDepth,
	const float* __restrict__ prevDepth,
	glm::mat4 reprojectionMatrix,
	float blendFactor,
	int width,
	int height,
	int frameNumber
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= width * height) return;

	int x = idx % width;
	int y = idx / width;
	glm::vec3 color = currentColor[idx];	
	float depth = currentDepth[idx];

	// Temporal blending
	if (frameNumber > 0) {
		// Convert to clip space
		glm::vec4 clipPos(
			(x + 0.5f) / width * 2.0f - 1.0f,
			(y + 0.5f) / height * 2.0f - 1.0f,
			depth,
			1.0f
		);

		// Reproject previous position
		glm::vec4 prevPos = reprojectionMatrix * clipPos;
		prevPos /= prevPos.w;

		// Convert to pixel coordinates
		float px = (prevPos.x * 0.5f + 0.5f) * width - 0.5f;
		float py = (prevPos.y * 0.5f + 0.5f) * height - 0.5f;
		//int prevIdx = static_cast<int>(py) * width + static_cast<int>(px);

		if (px >= 0 && px < width && py >= 0 && py < height) {

			glm::vec4 cd = bilinearSampleCombined(prevColor, prevDepth, width, height, px, py);
			glm::vec3 nMin, nMax;
			computeNeighborhoodMinMax(currentColor, x, y, width, height, nMin, nMax);

			glm::vec3 historyColor = glm::clamp(glm::vec3(cd), nMin - 0.05f, nMax + 0.05f);

			float localBlendFactor = blendFactor;
			color = glm::mix(color, historyColor, localBlendFactor);
		}

	}

	// Update history and write output
	int hist_idx = idx * 3;
	currentColor[idx] = color;
	//currentDepth[idx] = depth;

	color = glm::clamp(color, 0.0f, 1.0f) * 255.0f;
	color = glm::round(color);
	image[hist_idx] = static_cast<uint8_t>(color.x);
	image[hist_idx + 1] = static_cast<uint8_t>(color.y);
	image[hist_idx + 2] = static_cast<uint8_t>(color.z);
}

__global__ inline void clearBuffers(
	glm::vec3* __restrict__ currentColorBuffer,
	float* __restrict__ currentDepthBuffer,
	int width,
	int height,
	glm::vec3 clearColor,
	float clearDepth
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= width * height) return;

	currentColorBuffer[idx] = clearColor;
	currentDepthBuffer[idx] = clearDepth;
}