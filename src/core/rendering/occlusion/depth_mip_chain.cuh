#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include "config.h"

// Lightweight POD passed to kernels
struct DepthMipChainDevice {
	float* d_mip;
	int    width;
	int    height;
	int    levels;
	size_t mip_size;

	static constexpr int MAX_LEVELS = 16;
	int    level_widths[MAX_LEVELS];
	int    level_heights[MAX_LEVELS];
	size_t level_offsets[MAX_LEVELS];

	__host__ void precompute() {
		for (int l = 0; l < levels; ++l) {
			level_widths[l] = level_width(width, levels, l);
			level_heights[l] = level_height(height, levels, l);
			level_offsets[l] = level_offset(width, height, levels, l);
		}
	}

	__host__ __device__ __forceinline__ size_t index_of(int x, int y, int l) const {
		return level_offsets[l] + y * level_widths[l] + x;
	}

	__device__ static inline uint32_t morton2D(uint32_t x, uint32_t y) {
		x = (x | (x << 8)) & 0x00FF00FF;
		x = (x | (x << 4)) & 0x0F0F0F0F;
		x = (x | (x << 2)) & 0x33333333;
		x = (x | (x << 1)) & 0x55555555;

		y = (y | (y << 8)) & 0x00FF00FF;
		y = (y | (y << 4)) & 0x0F0F0F0F;
		y = (y | (y << 2)) & 0x33333333;
		y = (y | (y << 1)) & 0x55555555;

		return x | (y << 1);
	}

	__host__ __device__ static int compute_levels(int width, int height) {
		int maxDim = glm::max(width, height);
		int finestExp = 0;
		while ((1 << finestExp) < maxDim) ++finestExp;
		return finestExp + 1;
	}

	__host__ __device__ static int level_width(int width, int levels, int l) {
		int finestExp = levels - 1;
		return glm::max(1, (width + (1 << (finestExp - l)) - 1) >> (finestExp - l));
	}

	__host__ __device__ static int level_height(int height, int levels, int l) {
		int finestExp = levels - 1;
		return glm::max(1, (height + (1 << (finestExp - l)) - 1) >> (finestExp - l));
	}

	__host__ __device__ static size_t level_offset(int width, int height, int levels, int l) {
		size_t offset = 0;
		for (int i = 0; i < l; ++i)
			offset += (size_t)level_width(width, levels, i) * (size_t)level_height(height, levels, i);
		return offset;
	}
};

class DepthMipChain {
public:
	static constexpr int MAX_DEPTH = 9;

	DepthMipChain(int w, int h);
	~DepthMipChain();

	void build_mipchain(float* d_depth_buffer, int K_min_levels) const;

	DepthMipChainDevice device_view() const {
		return view;
	}

	float* get_device_mip_chain_pointer() const { return view.d_mip; }


	DepthMipChainDevice view;
};

#ifdef USE_TREE_TRAVERSAL



__forceinline__ __device__ bool depth_mip_is_occluded(const DepthMipChainDevice chain,
	float aabbDepth,
	glm::ivec2 minPix,
	glm::ivec2 maxPix) {

	//if (minPix.x > maxPix.x) { int t = minPix.x; minPix.x = maxPix.x; maxPix.x = t; }
	//if (minPix.y > maxPix.y) { int t = minPix.y; minPix.y = maxPix.y; maxPix.y = t; }
	//minPix.x = glm::max(minPix.x, 0); minPix.y = glm::max(minPix.y, 0);
	//maxPix.x = glm::min(maxPix.x, chain.width - 1); maxPix.y = glm::min(maxPix.y, chain.height - 1);

	struct Tile { int x, y, l; };
	constexpr size_t FRONTIER_BUDGET = 128;
	Tile frontier[FRONTIER_BUDGET];
	int frontierCount = 0;

	frontier[frontierCount++] = { 0, 0, 0 };

	const int maxTraversalLevel = glm::min(chain.levels - 1, DepthMipChain::MAX_DEPTH);

	while (frontierCount > 0) {
		Tile t = frontier[--frontierCount];

		int wl = DepthMipChainDevice::level_width(chain.width, chain.levels, t.l);
		int hl = DepthMipChainDevice::level_height(chain.height, chain.levels, t.l);

		//convert mip level coordinates to screenspace to compare to bounds
		//int tileW = (chain.width + wl - 1) / wl;
		//int tileH = (chain.height + hl - 1) / hl;

		int tileW = glm::min(chain.width, 1 << (chain.levels - t.l - 1));
		int tileH = glm::min(chain.height, 1 << (chain.levels - t.l - 1));

		int tileMinX = t.x * tileW;
		int tileMinY = t.y * tileH;
		int tileMaxX = glm::min(tileMinX + tileW - 1, chain.width - 1);
		int tileMaxY = glm::min(tileMinY + tileH - 1, chain.height - 1);

		//doesn't intersect the querried bounds
		if (tileMaxX < minPix.x || tileMaxY < minPix.y ||
			tileMinX > maxPix.x || tileMinY > maxPix.y) {
			continue;
		}

		size_t idx = DepthMipChainDevice::index_of(t.x, t.y, t.l, wl,
			chain.width, chain.height, chain.levels);
		float maxD = chain.d_mip[idx];

		//this tile occludes the gaussian
		//TODO: check if tile is fully further than e.g. 3 std from gaussian's mean
		if (maxD < aabbDepth) {
			continue;
		}

		//if at max depth, and gaussian is not occluded, we accept the gaussian.
		if (t.l >= maxTraversalLevel) {
			return false;
		}

		// Subdivide search
		int cx = t.x * 2, cy = t.y * 2;
		int wlChild = DepthMipChainDevice::level_width(chain.width, chain.levels, t.l + 1);
		int hlChild = DepthMipChainDevice::level_height(chain.height, chain.levels, t.l + 1);

		for (int dy = 0; dy < 2; ++dy)
			for (int dx = 0; dx < 2; ++dx) {
				int nx = cx + dx, ny = cy + dy;
				if (nx < wlChild && ny < hlChild && frontierCount < FRONTIER_BUDGET) {
					frontier[frontierCount++] = { nx, ny, t.l + 1 };
				}
			}
	}

	return true;
}
#else

__forceinline__ __device__ bool depth_mip_is_occluded(const DepthMipChainDevice chain,
	float aabbDepth,
	glm::ivec2 minPix,
	glm::ivec2 maxPix) {

	minPix.x = glm::max(0, minPix.x);
	minPix.y = glm::max(0, minPix.y);
	maxPix.x = glm::min(chain.width - 1, maxPix.x);
	maxPix.y = glm::min(chain.height - 1, maxPix.y);

	int w = maxPix.x - minPix.x + 1;
	int h = maxPix.y - minPix.y + 1;
	int d = glm::max(w, h);

	int lodFromSize = (int)ceilf(log2f((float)d / 2.0f));
	lodFromSize = glm::clamp(lodFromSize, 0, chain.levels - 1);
	int l = (chain.levels - 1) - lodFromSize;

	const int wl = chain.level_widths[l];
	const int hl = chain.level_heights[l];

	int tileW = glm::max(1, glm::min(chain.width, 1 << (chain.levels - l - 1)));
	int tileH = glm::max(1, glm::min(chain.height, 1 << (chain.levels - l - 1)));

	//tile coordinates, premultiplied by wl for index
	int tx0 = glm::max(0, glm::min(wl - 1, minPix.x / tileW));
	int ty0 = glm::max(0, glm::min(hl - 1, minPix.y / tileH)) * wl;
	int tx1 = glm::max(0, glm::min(wl - 1, maxPix.x / tileW));
	int ty1 = glm::max(0, glm::min(hl - 1, maxPix.y / tileH)) * wl;

	size_t base_pointer = chain.level_offsets[l];

	float zMax = fmaxf(
		chain.d_mip[base_pointer + tx0 + ty0],
		fmaxf(
			chain.d_mip[base_pointer + tx1 + ty0],
			fmaxf(
				chain.d_mip[base_pointer + tx0 + ty1],
				chain.d_mip[base_pointer + tx1 + ty1]
			)));


	return (zMax < aabbDepth);
}

#endif // USE_TREE_TRAVERSAL

// Kernels
__global__ void blit_fullres_to_finest_kernel(const DepthMipChainDevice chain,
	const float* __restrict__ d_src);

__global__ void reduce_level_max_kernel(const DepthMipChainDevice chain,
	int l);
