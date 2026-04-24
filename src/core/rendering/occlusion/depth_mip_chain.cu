#include "depth_mip_chain.cuh"
#include "utils/utils.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>

__global__ void blit_fullres_to_finest_kernel(const DepthMipChainDevice chain,
    const float* __restrict__ d_src) {
    const int finestL = chain.levels - 1;
    const int wl = chain.level_widths[finestL];
    const int hl = chain.level_heights[finestL];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= wl || y >= hl) return;

    const size_t dstIdx = chain.index_of(x, y, finestL);
    const size_t srcIdx = (size_t)y * (size_t)wl + (size_t)x;
    if (dstIdx >= chain.mip_size || srcIdx >= (size_t)wl * (size_t)hl) return;

    chain.d_mip[dstIdx] = d_src[srcIdx];   
}

struct ReduceMax {
    __device__ float init() const { return -INFINITY; }
    __device__ float op(float a, float b) const { return fmaxf(a, b); }
};

struct ReduceMin {
    __device__ float init() const { return INFINITY; }
    __device__ float op(float a, float b) const { return fminf(a, b); }
};


template <typename Reducer>
__global__ void reduce_level_kernel(const DepthMipChainDevice chain, int l, Reducer reducer) {
    const int wl = chain.level_widths[l];
    const int hl = chain.level_heights[l];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= wl || y >= hl) return;

    const int wl_f = chain.level_widths[l + 1];
    const int hl_f = chain.level_heights[l + 1];

    const int fx0 = x * 2;
    const int fy0 = y * 2;

    float v = reducer.init();

    // gather children in l+1
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            const int fx = fx0 + dx;
            const int fy = fy0 + dy;
            if (fx < wl_f && fy < hl_f) {
                const size_t cIdx = chain.index_of(fx, fy, l + 1);
                v = reducer.op(v, chain.d_mip[cIdx]);
            }
        }
    }

    const size_t dstIdx = chain.index_of(x, y, l);
    if (dstIdx >= chain.mip_size) return;
    chain.d_mip[dstIdx] = v;
}


DepthMipChain::DepthMipChain(int w, int h) {
    view.width = w;
    view.height = h;

    view.levels = DepthMipChainDevice::compute_levels(view.width, view.height);

    view.mip_size = 0;
    for (int l = 0; l < view.levels; ++l) {
        view.mip_size += (size_t)DepthMipChainDevice::level_width(view.width, view.levels, l)
            * (size_t)DepthMipChainDevice::level_height(view.height, view.levels, l);
    }

    ERRCHECK(cudaMalloc((void**) &view.d_mip, view.mip_size * sizeof(float)));

//#ifndef NDEBUG
//    // Optional debug dump of mip layout
//    for (int i = 0; i < levels; ++i) {
//        int wl = DepthMipChainDevice::level_width(width, levels, i);
//        int hl = DepthMipChainDevice::level_height(height, levels, i);
//        std::cout << "DepthMipChain level " << i
//            << " wl=" << wl << " hl=" << hl
//            << " index0=" << DepthMipChainDevice::index_of(0, 0, i, wl, width, height, levels)
//            << " indexLast=" << DepthMipChainDevice::index_of(wl - 1, hl - 1, i, wl, width, height, levels)
//            << std::endl;
//    }
//#endif

   
    view.precompute();
}

DepthMipChain::~DepthMipChain() {
    if (view.d_mip) {
        ERRCHECK(cudaFree(view.d_mip));
    }
}

void DepthMipChain::build_mipchain(float* d_depth_buffer, int K_min_levels) const {
    const DepthMipChainDevice chain = device_view();

    {
        const int finestL = chain.levels - 1;
        const int wl = DepthMipChainDevice::level_width(chain.width, chain.levels, finestL);
        const int hl = DepthMipChainDevice::level_height(chain.height, chain.levels, finestL);

        dim3 block(16, 16);
        dim3 grid((wl + block.x - 1) / block.x, (hl + block.y - 1) / block.y);

        KERNEL_LAUNCH(blit_fullres_to_finest_kernel, grid, block, chain, d_depth_buffer);
    }

    for (int l = chain.levels - 2; l >= 0; --l) {
        const int wl = DepthMipChainDevice::level_width(chain.width, chain.levels, l);
        const int hl = DepthMipChainDevice::level_height(chain.height, chain.levels, l);

        dim3 block(16, 16);
        dim3 grid((wl + block.x - 1) / block.x, (hl + block.y - 1) / block.y);

        int idx_from_finest = (chain.levels - 2) - l;
        bool useMin = idx_from_finest < K_min_levels;

        if (useMin) {
            ReduceMin reducer;
            KERNEL_LAUNCH((reduce_level_kernel<ReduceMin>), grid, block, chain, l, reducer);
        }
        else {
            ReduceMax reducer;
            KERNEL_LAUNCH((reduce_level_kernel<ReduceMax>), grid, block, chain, l, reducer);
        }
    }

#ifndef NDEBUG
    // In debug builds, force sync to catch errors early
    ERRCHECK(cudaDeviceSynchronize());
#endif
}