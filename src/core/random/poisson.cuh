#pragma once
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "random.cuh"
#include "config.h"

namespace Random {

    // Poisson sampler using Giles' QN3 Normal asymptotic approximation
    // Giles, M.B. (2016), Algorithm 955, ACM TOMS 42(1)
    __device__ __forceinline__ uint32_t poisson(uint2& state, float lambda)
    {
        float2 u = pcg2d(state);

        float w = boxMuller(u.x, u.y).x;
        float w2 = w * w;
        float w3 = w2 * w;
        float w4 = w2 * w2;

        float s = __fsqrt_rn(lambda);   // sqrt(lambda)
        float inv_s = 1.0f / s;             // lambda^{-1/2}
        float inv_l = 1.0f / lambda;        // lambda^{-1}

        // QN1: lambda + sqrt(lambda)*w + (w^2 - 1) / 6
        float kf = lambda + s * w + (w2 - 1.0f) / 6.0f;

        // QN2 term: lambda^{-1/2} * ( -1/36 * w - 1/72 * w^3 )
        kf += inv_s * (-(1.0f / 36.0f) * w - (1.0f / 72.0f) * w3);

        // QN3 term: lambda^{-1} * ( -8/405 + 7/810 * w^2 + 1/270 * w^4 )
        kf += inv_l * (-(8.0f / 405.0f) + (7.0f / 810.0f) * w2 + (1.0f / 270.0f) * w4);

        int ki = __float2int_rn(kf);
        uint32_t k = (uint32_t)max(ki, 0);

        return k;
    }

} // namespace Random
