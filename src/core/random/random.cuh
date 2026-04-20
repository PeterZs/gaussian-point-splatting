#pragma once
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <queue>
#include <numeric>
#include <math_constants.h>


namespace Random {
#ifndef RECIPROCAL_2e32
#define RECIPROCAL_2e32 2.3283064365387e-10f
#endif // !RECIPROCAL_2e32

	__device__ __forceinline__ unsigned int TausStep(unsigned int z, int S1, int S2, int S3, unsigned int M) {
		unsigned int b = (((z << S1) ^ z) >> S2);
		return ((z & M) << S3) ^ b;
	}

	__device__ __forceinline__ unsigned int LCGStep(unsigned int z, unsigned int A, unsigned int C) {
		return A * z + C;
	}

	__device__ __forceinline__ uint4 initState(int seed, int idx) {
		uint4 state;
		state.x = seed ^ idx;
		state.y = seed ^ (idx * 6364136223846793005ULL);
		state.z = seed ^ (idx * 1442695040888963407ULL);
		state.w = seed ^ (idx * 5027142804233777093ULL);
		return state;
	}

	__device__ __forceinline__ uint3 initState3(int seed, int idx) {
		uint3 state;
		state.x = seed ^ idx;
		state.y = seed ^ (idx * 6364136223846793005ULL);
		state.z = seed ^ (idx * 1442695040888963407ULL);
		return state;
	}

	__device__ __forceinline__ float hybridTaus(uint4& state) {
		state.x = TausStep(state.x, 13, 19, 12, 4294967294);
		state.y = TausStep(state.y, 2, 25, 4, 4294967288);
		state.z = TausStep(state.z, 3, 11, 17, 4294967280);
		state.w = LCGStep(state.w, 1664525, 1013904223);

		return 2.3283064365387e-10 * (state.x ^ state.y ^ state.z ^ state.w);
	}

	__device__ __forceinline__ float compute_box_muller_lower_bound(const float sigma_truncation = 3.0f) {
		return __expf(-0.5f * (sigma_truncation * sigma_truncation));
	}

	__device__ __host__ __forceinline__ float rescale_uniform(float u, float lower_bound) {
		//return u;
		return lower_bound + (1 - lower_bound) * u;
	}

	//__device__ __forceinline__ float2 boxMuller(float u1, float u2) {
	//    //return { u1 - 0.5f, u2 - 0.5f };
	//    float R = sqrtf(-2.0f * logf(u1));
	//    float theta = 6.2831853f * u2;
	//    float z0 = R * cosf(theta);
	//    float z1 = R * sinf(theta);
	//    return make_float2(z0, z1);
	//}

	__device__ __forceinline__ float2 boxMuller(float u1, float u2) {
		float R = __fsqrt_rn(fmaxf(-2.0f * __logf(glm::clamp(u1, 1.e-37f, 1.0f)), 0.0f));
		float theta = 2.0f * CUDART_PI_F * u2;
		float s, c;
		__sincosf(theta, &s, &c);
		return make_float2(R * c, R * s);
	}

	//__device__ __forceinline__ float dilog(float x) {
	//	float s = 1.0f - x;
	//	float y = fmaf(fmaf(fmaf(fmaf(fmaf(fmaf(fmaf(-1.068681974f, x, 3.334685126f), x, -4.173996483f), x, 2.567860600f), x, -0.884150470f), x, -0.123550674f), x, 1.992765336f), x, 6.74195669e-05f);
	//	y += (s > 0.0f) ? (s * __logf(glm::clamp(s, 1.e-37f, 1.0f))) : 0.0f;
	//	return y;
	//}

	//__device__ __forceinline__ float inv_dilog(float x) {
	//	return fmaf(fmaf(fmaf(fmaf(fmaf(fmaf(fmaf(fmaf(fmaf(fmaf(-0.322192871f, x, 2.562830719f), x, -8.669815892f), x, 16.243561492f), x, -18.394710417f), x, 12.891925084f), x, -5.501796628f), x, 1.361194673f), x, -0.417517201f), x, 1.008071981f), x, -6.39879974e-05f);
	//}

	__device__ __forceinline__ float dilog(float x)
	{
		float y = -6.09201442e-01f;
		y = fmaf(y, x, 1.79126616e+00f);
		y = fmaf(y, x, -2.14953223e+00f);
		y = fmaf(y, x, 1.26304372e+00f);
		y = fmaf(y, x, -4.59069895e-01f);
		y = fmaf(y, x, -1.87417414e-01f);
		y = fmaf(y, x, 1.99603130e+00f);
		y = fmaf(y, x, 5.99669467e-05f);

		float s = 1.0f - x;
		if (s > 0.0f)
			y += s * __logf(fmaxf(s, 1e-37f));

		return y;
	}


	// Li2(1) = pi^2 / 6
#define LI2_MAX 1.6449340668482264f

	__device__ __forceinline__ float inv_dilog(float x)
	{
		float t = fminf(x / LI2_MAX, 1.0f);

		float y = -1.27463503e+01f;
		y = fmaf(y, t, 5.88993459e+01f);
		y = fmaf(y, t, -1.16025780e+02f);
		y = fmaf(y, t, 1.26945827e+02f);
		y = fmaf(y, t, -8.43108826e+01f);
		y = fmaf(y, t, 3.48799862e+01f);
		y = fmaf(y, t, -8.89606235e+00f);
		y = fmaf(y, t, 1.38640936e+00f);
		y = fmaf(y, t, -7.80640876e-01f);
		y = fmaf(y, t, 1.64841888e+00f);
		y = fmaf(y, t, -2.82836687e-05f);

		return y;
	}




	__device__ __forceinline__ float2 correctedBoxMuller(float u1, float u2, float alpha) {
		float a = 1.0f / fmaxf(alpha, 1.e-37f) * inv_dilog((1.0f - u1) * dilog(alpha));
		a = glm::clamp(a, 1.e-37f, 1.0f);
		float R = __fsqrt_rn(fmaxf(-2.0f * __logf(a), 0.0f));

		float theta = 2.0f * CUDART_PI_F * u2;
		float s, c;
		__sincosf(theta, &s, &c);
		return make_float2(R * c, R * s);
	}

	__device__ __forceinline__ uint3 pcg3d(uint3 v) {
		v.x = v.x * 1664525u + 1013904223u;
		v.y = v.y * 1664525u + 1013904223u;
		v.z = v.z * 1664525u + 1013904223u;

		v.x += v.y * v.z;
		v.y += v.z * v.x;
		v.z += v.x * v.y;

		v.x ^= v.x >> 16u;
		v.y ^= v.y >> 16u;
		v.z ^= v.z >> 16u;

		v.x += v.y * v.z;
		v.y += v.z * v.x;
		v.z += v.x * v.y;
		return v;
	}

	__device__ __forceinline__ uint2 pcg2d_uint(uint2 seed) {
		seed.x = seed.x * 1664525u + 1013904223u;
		seed.y = seed.y * 1664525u + 1013904223u;

		seed.x += seed.y * 1664525u;
		seed.y += seed.x * 1664525u;

		seed.x ^= seed.x >> 16u;
		seed.y ^= seed.y >> 16u;

		seed.x += seed.y * 1664525u;
		seed.y += seed.x * 1664525u;

		seed.x ^= seed.x >> 16u;
		seed.y ^= seed.y >> 16u;
		return seed;
	}


	__device__ __forceinline__ float2 pcg2d(uint2& seed) {
		seed = pcg2d_uint(seed);
		return { seed.x * RECIPROCAL_2e32, seed.y * RECIPROCAL_2e32 };
	}

	__device__ __forceinline__ int stochastic_round(float x, float u) {
		int base = static_cast<int>(floorf(x));
		float frac = x - static_cast<float>(base);
		return glm::max(0, (u < frac) ? (base + 1) : base);
	}

	__device__ __forceinline__ uint32_t hash32(uint32_t x) {
		x ^= x >> 16;
		x *= 0x7feb352d;
		x ^= x >> 15;
		x *= 0x846ca68b;
		x ^= x >> 16;
		return x;
	}

	__device__ __forceinline__ uint2 makeSeed(uint32_t idx, uint32_t frame) {

		const uint32_t C1 = 0x9E3779B9u;
		const uint32_t C2 = 0xBB67AE85u;
		const uint32_t C3 = 0x3C6EF372u;
		const uint32_t C4 = 0xA54FF53Au;

		uint32_t h1 = Random::hash32(idx ^ (frame * C1));
		uint32_t h2 = Random::hash32((idx * C2) ^ frame);

		h1 ^= (h2 * C3);
		h2 ^= (h1 * C4);

		return make_uint2(h1, h2);
	}

	template<int d>
	__device__ __forceinline__ int icdfGaussianBinSampling(float u, float x, float std) {
		float w[d * 2 + 1];
		float s = 0;
		float r_2v = -1.0f / (2.0f * std * std);
		x = x - truncf(x);
		for (int i = -d; i <= d; i++)
		{
			float delta = (x - float(i) - 0.5f);
			float g = __expf(r_2v * delta * delta);
			s += g;
			w[d + i] = g;
		}

		float threshold = u * s;
		float cdf = 0.0f;
		for (int i = 0; i < 2 * d + 1; ++i) {
			cdf += w[i];
			if (cdf >= threshold) {
				return i - d;
			}
		}
		return d;
	}

}  // namespace Random
