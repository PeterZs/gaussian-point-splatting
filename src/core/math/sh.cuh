#pragma once
#include <vector_types.h>
#include <vector>
#include <vector_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <cmath>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <cuda_bf16.h>
#include <core/math/linalg.cuh>


struct SimpleCompactSH {
	SimpleCompactSH(const float _sh[48]) {
		for (int i = 0; i < 48; ++i) {
			sh[i] = nv_bfloat16(float(_sh[i]));
		}
	}

	nv_bfloat16 sh[48];
};

struct CompactSH {
	CompactSH(const float _sh[48], const glm::vec3* minSH, const glm::vec3* maxSH) {
		base = glm::vec3(_sh[0], _sh[1], _sh[2]);

		for (int i = 0; i < 15; ++i) {
			glm::vec3 SH = glm::vec3(_sh[3 * i + 3], _sh[3 * i + 4], _sh[3 * i + 5]);// / (base + 1e-8f);
			SH = (SH - minSH[i]) / (maxSH[i] - minSH[i]);
			sh[3 * i] = static_cast<uint8_t>(glm::clamp(SH.x * 255, 0.0f, 255.0f));
			sh[3 * i + 1] = static_cast<uint8_t>(glm::clamp(SH.y * 255, 0.0f, 255.0f));
			sh[3 * i + 2] = static_cast<uint8_t>(glm::clamp(SH.z * 255, 0.0f, 255.0f));
		}
	}

	__host__ __device__ __forceinline__ glm::vec3 get(int idx, const glm::vec3* minSH, const glm::vec3* maxSH) const {
		if (idx <= 0) return base;
		--idx;
		constexpr float r = 1.0f / 255.0f;
		glm::vec3 vec = glm::vec3(sh[3 * idx] * r, sh[3 * idx + 1] * r, sh[3 * idx + 2] * r);
		return glm::mix(minSH[idx], maxSH[idx], vec);// *(base + 1e-8f);
	}

	glm::vec3 base; //store in float
	uint8_t sh[45]; //store quantized
};

namespace CudaMath {

	__host__ __device__ __forceinline__ void eval_sh_sloan_l3(float out_shs[16], const glm::vec3 dir) {
		float x, y, z, z2, c0, s0, c1, s1, d, a, b;
		x = dir[0];
		y = dir[1];
		z = dir[2];
		z2 = z * z;
		d = 0.282094792f;
		out_shs[0] = d;
		d = 0.488602512f * z;
		out_shs[2] = d;
		a = z2 - 0.333333333f;
		d = 0.946174696f * a;
		out_shs[6] = d;
		b = z * (a - 0.266666667f);
		d = 1.865881663f * b;
		out_shs[12] = d;
		c1 = x;
		s1 = y;
		d = -0.488602512f;
		out_shs[1] = s1 * d;
		out_shs[3] = c1 * d;
		d = -1.092548431f * z;
		out_shs[5] = s1 * d;
		out_shs[7] = c1 * d;
		a = z2 - 0.2f;
		d = -2.285228997f * a;
		out_shs[11] = s1 * d;
		out_shs[13] = c1 * d;
		c0 = x * c1 - y * s1;
		s0 = y * c1 + x * s1;
		d = 0.546274215f;
		out_shs[4] = s0 * d;
		out_shs[8] = c0 * d;
		d = 1.445305721f * z;
		out_shs[10] = s0 * d;
		out_shs[14] = c0 * d;
		c1 = x * c0 - y * s0;
		s1 = y * c0 + x * s0;
		d = -0.59004359f;
		out_shs[9] = s1 * d;
		out_shs[15] = c1 * d;
	}

	/// <summary>
	/// Converts a color in [0,1] to 4 quantized bytes
	/// </summary>
	/// <param name="rgba"></param>
	/// <returns></returns>
	__host__ __device__ __forceinline__ static uint32_t vec4ToUintRGBA(const glm::vec4& rgba) {
		return
			(static_cast<uint8_t>(fminf(fmaxf(255.0f * rgba.x, 0.0f), 255.0f)) << 24) |
			(static_cast<uint8_t>(fminf(fmaxf(255.0f * rgba.y, 0.0f), 255.0f)) << 16) |
			(static_cast<uint8_t>(fminf(fmaxf(255.0f * rgba.z, 0.0f), 255.0f)) << 8) |
			(static_cast<uint8_t>(fminf(fmaxf(255.0f * rgba.w, 0.0f), 255.0f)));
	}

	__host__ __device__ __forceinline__ static glm::vec4 uintRGBAToVec4(const uint32_t rgba) {
		const uint8_t r = static_cast<uint8_t>((rgba >> 24) & 0xFFu);
		const uint8_t g = static_cast<uint8_t>((rgba >> 16) & 0xFFu);
		const uint8_t b = static_cast<uint8_t>((rgba >> 8) & 0xFFu);
		const uint8_t a = static_cast<uint8_t>((rgba >> 0) & 0xFFu);

		const float inv255 = 1.0f / 255.0f;
		return glm::vec4(r * inv255, g * inv255, b * inv255, a * inv255);
	}


	//__host__ __device__ __forceinline__ 

	__host__ __device__ __forceinline__ static uint64_t vec3To36Bits(const glm::vec3& rgb) {
		const int channel_size = 12;
		const int max_channel_size = 8;
		constexpr float a = (1 << max_channel_size) - 1.f;
		constexpr float b = 1 << (channel_size - max_channel_size);
		return
			((static_cast<uint64_t>(rgb.x * a) & 0xFFF) << 2 * channel_size) |
			((static_cast<uint64_t>(rgb.y * a) & 0xFFF) << channel_size) |
			((static_cast<uint64_t>(rgb.z * a) & 0xFFF));
	}
#define CHANNELS 3

	// Scalar constants as constexpr
	constexpr float SH_C0 = 0.28209479177387814f;
	constexpr float SH_C1 = 0.4886025119029199f;

	// Array constants as static __device__ to avoid redefinition
	__device__ static const float SH_C2[] = {
		1.0925484305920792f,
		-1.0925484305920792f,
		0.31539156525252005f,
		-1.0925484305920792f,
		0.5462742152960396f
	};

	__device__ static const float SH_C3[] = {
		-0.5900435899266435f,
		2.890611442640554f,
		-0.4570457994644658f,
		0.3731763325901154f,
		-0.4570457994644658f,
		1.445305721320277f,
		-0.5900435899266435f
	};

	__device__ static const float SH_C4[] = {
		2.5033429417967046f,
		-1.7701307697799304f,
		0.9461746957575601f,
		-0.6690465435572892f,
		0.10578554691520431f,
		-0.6690465435572892f,
		0.47308734787878004f,
		-1.7701307697799304f,
		0.6258357354491761f
	};

	// Inline function definitions
	__host__ __device__ inline unsigned num_sh_bases(const unsigned degree) {
		if (degree == 0) return 1;
		if (degree == 1) return 4;
		if (degree == 2) return 9;
		if (degree == 3) return 16;
		return 25; // degree >=4
	}

	__device__ __host__ __forceinline__ glm::vec3 color_from_base(const float* shs) {
		return SH_C0 * glm::vec3(shs[0], shs[1], shs[2]) + 0.5f;
	}

	__device__ __host__ __forceinline__ void base_from_color(const glm::vec3 color, float* shs) {
		glm::vec3 out = (color - 0.5f) / SH_C0;
		shs[0] = out.x;
		shs[1] = out.y;
		shs[2] = out.z;
	}

	__device__ inline glm::vec3 computeColorFromSHFloat(int deg, const glm::vec3 pos, const glm::vec3 campos, const float* shs) {
		glm::vec3 dir = pos - campos;
		dir = glm::normalize(dir);

#define SH(i) glm::vec3(shs[3 * i], shs[3 * i + 1], shs[3 * i + 2])

		glm::vec3 result = SH_C0 * SH(0);
		if (deg > 0) {
			float x = dir.x, y = dir.y, z = dir.z;
			result = result - SH_C1 * y * SH(1) + SH_C1 * z * SH(2) - SH_C1 * x * SH(3);
			if (deg > 1) {
				float xx = x * x, yy = y * y, zz = z * z;
				float xy = x * y, yz = y * z, xz = x * z;
				result += SH_C2[0] * xy * SH(4) +
					SH_C2[1] * yz * SH(5) +
					SH_C2[2] * (2.0f * zz - xx - yy) * SH(6) +
					SH_C2[3] * xz * SH(7) +
					SH_C2[4] * (xx - yy) * SH(8);
				if (deg > 2) {
					result += SH_C3[0] * y * (3.0f * xx - yy) * SH(9) +
						SH_C3[1] * xy * z * SH(10) +
						SH_C3[2] * y * (4.0f * zz - xx - yy) * SH(11) +
						SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * SH(12) +
						SH_C3[4] * x * (4.0f * zz - xx - yy) * SH(13) +
						SH_C3[5] * z * (xx - yy) * SH(14) +
						SH_C3[6] * x * (xx - 3.0f * yy) * SH(15);
				}
			}
		}
		result += 0.5f;
		return glm::max(result, glm::vec3(0.0f));
#undef SH
	}

	//__device__ inline glm::vec3 computeColorFromCompactSH(const int deg, const glm::vec3 pos, const glm::vec3 campos, const CompactSH& sh, const glm::vec3* minSH, const glm::vec3* maxSH) {
	//	glm::vec3 dir = pos - campos;
	//	dir = glm::normalize(dir);
	//	glm::vec3 result = SH_C0 * sh.get(0, minSH, maxSH);
	//	if (deg > 0) {
	//		float x = dir.x, y = dir.y, z = dir.z;
	//		result = result - SH_C1 * y * sh.get(1, minSH, maxSH) +
	//			SH_C1 * z * sh.get(2, minSH, maxSH) -
	//			SH_C1 * x * sh.get(3, minSH, maxSH);
	//		if (deg > 1) {
	//			float xx = x * x, yy = y * y, zz = z * z;
	//			float xy = x * y, yz = y * z, xz = x * z;
	//			result += SH_C2[0] * xy * sh.get(4, minSH, maxSH) +
	//				SH_C2[1] * yz * sh.get(5, minSH, maxSH) +
	//				SH_C2[2] * (2.0f * zz - xx - yy) * sh.get(6, minSH, maxSH) +
	//				SH_C2[3] * xz * sh.get(7, minSH, maxSH) +
	//				SH_C2[4] * (xx - yy) * sh.get(8, minSH, maxSH);
	//			if (deg > 2) {
	//				result += SH_C3[0] * y * (3.0f * xx - yy) * sh.get(9, minSH, maxSH) +
	//					SH_C3[1] * xy * z * sh.get(10, minSH, maxSH) +
	//					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh.get(11, minSH, maxSH) +
	//					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh.get(12, minSH, maxSH) +
	//					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh.get(13, minSH, maxSH) +
	//					SH_C3[5] * z * (xx - yy) * sh.get(14, minSH, maxSH) +
	//					SH_C3[6] * x * (xx - 3.0f * yy) * sh.get(15, minSH, maxSH);
	//			}
	//		}
	//	}
	//	result += 0.5f;
	//	return glm::max(result, glm::vec3(0.0f));
	//}

	__device__ inline glm::vec3 computeColorFromCompactSH(const int deg, const glm::vec3 pos, const glm::vec3 campos, const CompactSH& sh, const glm::vec3* minSH, const glm::vec3* maxSH) {
		glm::vec3 dir = pos - campos;
		dir = glm::normalize(dir);

		float eval_sh[16];
		eval_sh_sloan_l3(eval_sh, dir);

		glm::vec3 result(0.0f);
#pragma unroll
		for (int i = 0; i < 16; ++i) {
			const glm::vec3 c = sh.get(i, minSH, maxSH);
			result += eval_sh[i] * c;
		}

		result += 0.5f;
		return glm::max(result, glm::vec3(0.0f));
	}
}