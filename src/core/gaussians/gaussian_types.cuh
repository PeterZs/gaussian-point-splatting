#pragma once
#include "config.h"

#include <vector_types.h>
#include <vector>
#include <core/math/linalg.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <cuda_bf16.h>
#include <core/math/sh.cuh>
#include "utils/utils.cuh"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif // M_PI
#ifndef M_PI_CUBED
#define M_PI_CUBED 31.0062766803f
#endif // !M_PI_CUBED
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923f
#endif // !M_PI_2
#ifndef M_1_SQRT2
#define M_1_SQRT2 0.70710678118f
#endif // !M_1_SQRT2
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237f
#endif // !M_SQRT2
#ifndef M_EPS
#define M_EPS 1.0e-12f
#endif // !M_EPS

namespace CudaMath {
	//todo improve/make tighter
	__host__ __device__ __forceinline__ void aabb_from_gaussian(const glm::vec3 mean, const glm::mat3 covariance, glm::vec3& out_min, glm::vec3& out_max, const float std = 3.0f)
	{
		glm::vec3 extent = std * glm::sqrt(glm::vec3{ covariance[0].x, covariance[1].y, covariance[2].z });
		out_min = mean - extent;
		out_max = mean + extent;
	}

	//adhoc conversion of opacity trained using 3DGS to mass used by our renderer.
	__host__ __forceinline__ float opacity_to_mass(const glm::vec3 scale, const float opacity) {
		//float sqrt_det_geometric_mean = glm::max(scale.x, glm::max(scale.y, scale.z));
		//float sqrt_det_geometric_mean = glm::pow(scale.x * scale.y * scale.z, 1.0f / 3.0f);
		float sqrt_det_geometric_mean = glm::max(
			glm::pow(scale.x * scale.y, 1.0f / 2.0f),
			glm::max(glm::pow(scale.x * scale.z, 1.0f / 2.0f),
				glm::pow(scale.y * scale.z, 1.0f / 2.0f)));
		float mass = 2.0f * M_PI * opacity * sqrt_det_geometric_mean * sqrt_det_geometric_mean;
		return mass;
	}

	__host__ __device__ __forceinline__
		void aabb_from_gaussian(const glm::vec3& mean,
			const glm::vec3& scale,     // variance per axis
			const glm::quat& rotation, // unit quaternion
			glm::vec3& out_min,
			glm::vec3& out_max,
			float stddev_factor = 3.0f)
	{
		// convert variance -> stddev
		glm::vec3 sigma = stddev_factor * glm::sqrt(glm::abs(scale));

		// rotation matrix from quaternion
		glm::mat3 R = glm::mat3_cast(rotation);

		// build rotated scale matrix
		glm::mat3 M = R * glm::mat3(glm::vec3(sigma.x, 0, 0),
			glm::vec3(0, sigma.y, 0),
			glm::vec3(0, 0, sigma.z));

		// row lengths give half-extents
		glm::vec3 extent(
			glm::length(M[0]),  // row 0
			glm::length(M[1]),  // row 1
			glm::length(M[2])   // row 2
		);

		out_min = mean - extent;
		out_max = mean + extent;
	}
};


struct CPUGaussian {
	glm::vec3 mean; //mean
	//float3 normal;
#ifdef DISABLE_SPHERICAL_HARMONICS
	float sh[3]; //only store base sh coefficients
#else
	float sh[48]; //color constants + 3 * 15 SH constants until L3
#endif // ! DISABLE_SPHERICAL_HARMONICS	
	float opacity; //alpha
	glm::vec3 scale; //scale of gaussian
	glm::vec4 rotation; //quaternion representing rotation.
};


static CPUGaussian CreateGaussian(const glm::vec3 mean, const glm::vec3 scale, const glm::vec4 rotation, const uint32_t color) {
	constexpr float rsh0 = 1.0f / 0.28209479177387814f;

	float r = (((color >> 24) & 0xFF) / 255.0f - 0.5f) * rsh0;
	float g = (((color >> 16) & 0xFF) / 255.0f - 0.5f) * rsh0;
	float b = (((color >> 8) & 0xFF) / 255.0f - 0.5f) * rsh0;
	float a = (color & 0xFF) / 255.0f;

	return CPUGaussian{
		mean,
		{
			r,g,b,
#ifndef DISABLE_SPHERICAL_HARMONICS			
			0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
#endif
		},
		a,
		scale,
		rotation
	};
}

struct GaussianSceneBounds {
	//bounds are used for quantization
	glm::vec3 minPos, maxPos;
	glm::vec3 minScale, maxScale; //log encoded
	glm::vec3 minSH[15], maxSH[15]; //store bounds for each SH channel

	GaussianSceneBounds() = default;

	GaussianSceneBounds(const std::vector<CPUGaussian>& gaussians) {
		maxPos = glm::vec3(std::numeric_limits<float>::lowest());
		minPos = glm::vec3(std::numeric_limits<float>::max());
		maxScale = glm::vec3(std::numeric_limits<float>::lowest());
		minScale = glm::vec3(std::numeric_limits<float>::max());

		for (size_t i = 0; i < 15; i++)
		{
			maxSH[i] = glm::vec3(std::numeric_limits<float>::lowest());
			minSH[i] = glm::vec3(std::numeric_limits<float>::max());
		}

		for (size_t i = 0; i < gaussians.size(); ++i)
		{
			maxPos = glm::max(maxPos, gaussians[i].mean);
			minPos = glm::min(minPos, gaussians[i].mean);
			maxScale = glm::max(maxScale, glm::log(gaussians[i].scale));
			minScale = glm::min(minScale, glm::log(gaussians[i].scale));

			glm::vec3 baseSH = glm::vec3(gaussians[i].sh[0], gaussians[i].sh[1], gaussians[i].sh[2]);

			for (size_t j = 0; j < 15; ++j) {
				glm::vec3 SH = glm::vec3(gaussians[i].sh[3 * j + 3],
					gaussians[i].sh[3 * j + 4],
					gaussians[i].sh[3 * j + 5]);
				//SH /= (baseSH + 1e-8f);

				maxSH[j] = glm::max(maxSH[j], SH);
				minSH[j] = glm::min(minSH[j], SH);
			}
		}

		constexpr float SAFETY_MARGIN = 0.001f;
		maxPos += SAFETY_MARGIN;
		minPos -= SAFETY_MARGIN;
	}
};

struct QuantizedGaussianGeometry {
	//Position bit counts: 23,22,23
	//Rotation : 10 bits per channel (3)
	//SCale : 10 bits per channel (3)
	uint32_t word0;//word0: [ posX bits 0–22 ] [ posY bits 0–8 ]
	uint32_t word1;//word1: [ posY bits 9–21 ] [ posZ bits 0–9 ]
	uint32_t word2;//word2: [ posZ bits 10–22 ] [ rotX 10 bits ] [ rotY 10 bits ]
	uint32_t word3;//word3: [ rotZ 10 bits ] [ scaleX 10 bits ] [ scaleY 10 bits ] [ scaleZ 10 bits ]
};

//alternative unquantized gaussian. Used for evaluation
struct GaussianGeometry {
	glm::vec3 mean;
	glm::vec4 quat;
	glm::vec3 scale;
};


__host__ __device__ __forceinline__ float normalized_byte_to_float(const uint8_t value) {
	// Convert the normalized byte (0..255) to a normalized float (0..1)
	return static_cast<float>(value) / 255.0f;
}

__host__ __device__ __forceinline__ uint8_t normalized_float_to_byte(const float value) {
	// Clamp the input value to the range [0..1], then scale to the range [0..255]
	return static_cast<uint8_t>(fminf(fmaxf(value, 0.0f), 1.0f) * 255.0f + 0.5f);
}

//from  https://marc-b-reynolds.github.io/quaternions/2017/05/02/QuatQuantPart1.html
__device__ __host__ __forceinline__ glm::vec3 quat_fem(glm::vec4 q)
{
	float w = q.w;
	float a = 1.0f - w * w;
	float b = glm::inversesqrt(a + M_EPS);
	float k = a * b;
	float s = (2.0f / M_PI) * atan(k / w) * b;
	return s * glm::vec3(q);
}

#define M_EPS 1e-8f
#define HALF_PI 1.57079632679f

__host__ __device__ __forceinline__
glm::vec4 quat_iem(const glm::vec3 v)
{
	float d = v.x * v.x + v.y * v.y + v.z * v.z;

#if defined(__CUDA_ARCH__)
	// device path
	float t = rsqrtf(d + M_EPS);
	float a = HALF_PI * t * d;
	float s = __sinf(a);
	float c = __cosf(a);
#else
	// host path
	float t = 1.0f / std::sqrt(d + M_EPS);
	float a = HALF_PI * t * d;
	float s = std::sinf(a);
	float c = std::cosf(a);
#endif
	float k = s * t;
	return glm::vec4(k * v, c);
}

__host__ __device__ __forceinline__
QuantizedGaussianGeometry encode_gaussian_geometry(
	const glm::vec3& pos,
	const glm::vec4& quat,
	const glm::vec3& scale,
	const GaussianSceneBounds& scene)
{
	QuantizedGaussianGeometry qg;

	// -----------------------------
	// Position quantization
	// -----------------------------
	glm::vec3 t = glm::clamp(
		(pos - scene.minPos) / (scene.maxPos - scene.minPos),
		0.0f, 1.0f
	);

	uint32_t px = (uint32_t)(t.x * ((1u << 23) - 1u));
	uint32_t py = (uint32_t)(t.y * ((1u << 22) - 1u));
	uint32_t pz = (uint32_t)(t.z * ((1u << 23) - 1u));

	// -----------------------------
	// Rotation quantization (10 bits each)
	// -----------------------------
	glm::vec4 q = glm::normalize(quat);
	if (q.w < 0.0f) q = -q;

	glm::vec3 v = quat_fem(q); // [-1,1]
	glm::vec3 vr = glm::clamp((v + 1.0f) * 0.5f, 0.0f, 1.0f);

	uint32_t rx = (uint32_t)(vr.x * 1023.0f);
	uint32_t ry = (uint32_t)(vr.y * 1023.0f);
	uint32_t rz = (uint32_t)(vr.z * 1023.0f);

	// -----------------------------
	// Scale quantization (10 bits each)
	// -----------------------------
	glm::vec3 log_s = glm::log(scale);
	glm::vec3 st = glm::clamp(
		(log_s - scene.minScale) / (scene.maxScale - scene.minScale),
		0.0f, 1.0f
	);

	uint32_t sx = (uint32_t)(st.x * 1023.0f);
	uint32_t sy = (uint32_t)(st.y * 1023.0f);
	uint32_t sz = (uint32_t)(st.z * 1023.0f);

	// -----------------------------
	// Split fields for packing
	// -----------------------------
	uint32_t py_low9 = py & 0x1FFu;        // bits 0–8
	uint32_t py_high13 = (py >> 9) & 0x1FFFu;       // bits 9–21

	uint32_t pz_low19 = pz & 0x7FFFFu;      // bits 0–18
	uint32_t pz_high4 = (pz >> 19) & 0xFu;          // bits 19–22

	uint32_t rz_low8 = rz & 0xFFu;         // bits 0–7
	uint32_t rz_high2 = (rz >> 8) & 0x3u;          // bits 8–9

	// -----------------------------
	// Pack into 4 words
	// -----------------------------
	qg.word0 =
		(px & 0x7FFFFFu) |              // 23 bits
		(py_low9 << 23);                // 9 bits

	qg.word1 =
		(py_high13 & 0x1FFFu) |         // 13 bits
		(pz_low19 << 13);              // 19 bits

	qg.word2 =
		(pz_high4 & 0xFu) |  // 4 bits
		((rx & 0x3FFu) << 4) |  // 10 bits
		((ry & 0x3FFu) << 14) |  // 10 bits
		((rz_low8 & 0xFFu) << 24);      // 8 bits

	qg.word3 =
		(rz_high2 & 0x3u) |   // 2 bits
		((sx & 0x3FFu) << 2) |   // 10 bits
		((sy & 0x3FFu) << 12) |   // 10 bits
		((sz & 0x3FFu) << 22);          // 10 bits

	return qg;
}

__device__ __forceinline__
GaussianGeometry decode_gaussian_geometry(
	const QuantizedGaussianGeometry qg,
	const GaussianSceneBounds scene)
{
	GaussianGeometry result{};
	// -----------------------------
	// Unpack position
	// -----------------------------
	uint32_t px = qg.word0 & 0x7FFFFFu;      // 23 bits
	uint32_t py_low9 = (qg.word0 >> 23) & 0x1FFu;  // 9 bits

	uint32_t py_high13 = qg.word1 & 0x1FFFu; // 13 bits
	uint32_t pz_low19 = (qg.word1 >> 13) & 0x7FFFFu;// 19 bits

	uint32_t pz_high4 = qg.word2 & 0xFu;    // 4 bits

	uint32_t py = py_low9 | (py_high13 << 9);
	uint32_t pz = pz_low19 | (pz_high4 << 19);

	float fx = float(px) / float((1u << 23) - 1u);
	float fy = float(py) / float((1u << 22) - 1u);
	float fz = float(pz) / float((1u << 23) - 1u);

	result.mean = glm::mix(scene.minPos, scene.maxPos, glm::vec3(fx, fy, fz));

	// -----------------------------
	// Unpack rotation
	// -----------------------------
	uint32_t rx = (qg.word2 >> 4) & 0x3FFu;    // 10 bits
	uint32_t ry = (qg.word2 >> 14) & 0x3FFu;    // 10 bits
	uint32_t rz_low8 = (qg.word2 >> 24) & 0xFFu;     // 8 bits

	uint32_t rz_high2 = qg.word3 & 0x3u;      // 2 bits
	uint32_t rz = rz_low8 | (rz_high2 << 8);   // 10 bits

	glm::vec3 vr = glm::vec3(rx, ry, rz) / 1023.0f;
	glm::vec3 v = vr * 2.0f - 1.0f;

	result.quat = quat_iem(v);

	// -----------------------------
	// Unpack scale
	// -----------------------------
	uint32_t sx = (qg.word3 >> 2) & 0x3FFu;
	uint32_t sy = (qg.word3 >> 12) & 0x3FFu;
	uint32_t sz = (qg.word3 >> 22) & 0x3FFu;

	glm::vec3 st = glm::vec3(sx, sy, sz) / 1023.0f;
	result.scale = glm::mix(scene.minScale, scene.maxScale, st);
	result.scale = { __expf(result.scale.x), __expf(result.scale.y) , __expf(result.scale.z) };
	return result;
}

// --- Utility decode helpers ---

__host__ __device__ __forceinline__ float decode_opacity(uint8_t v) {
	// assumes you have normalized_byte_to_float
	extern __host__ __device__ float normalized_byte_to_float(uint8_t);
	return normalized_byte_to_float(v);
}
__host__ __device__ __forceinline__ float decode_opacity(float v) {
	return v;
}

// SCENES
#ifndef GAUSSIAN_COLOR_STORAGE_TYPE
#ifdef USE_SDR_COLOR
#define GAUSSIAN_COLOR_STORAGE_TYPE uint32_t
__host__ __device__ __forceinline__ GAUSSIAN_COLOR_STORAGE_TYPE encode_color(glm::vec3 color) {
	glm::vec3 c = glm::clamp(color, glm::vec3(0.0f), glm::vec3(4.0f));
	return (((uint32_t)(c.r * 255.f) & 0x3FF) << 20) |
		(((uint32_t)(c.g * 255.f) & 0x3FF) << 10) |
		((uint32_t)(c.b * 255.f) & 0x3FF);
}

__host__ __device__ __forceinline__ glm::vec3 decode_color(GAUSSIAN_COLOR_STORAGE_TYPE color) {
	return glm::vec3{
		((color >> 20) & 0x3FF) / 255.f,
		((color >> 10) & 0x3FF) / 255.f,
		((color) & 0x3FF) / 255.f
	};
}
#else
#define GAUSSIAN_COLOR_STORAGE_TYPE glm::vec3
__host__ __device__ __forceinline__ GAUSSIAN_COLOR_STORAGE_TYPE encode_color(glm::vec3 color) {
	return color;
}

__host__ __device__ __forceinline__ glm::vec3 decode_color(GAUSSIAN_COLOR_STORAGE_TYPE color) {
	return color;
}
#endif
#endif // GAUSSIAN_COLOR_STORAGE_TYPE

//========================================================================================================
// Compressed Scene
//========================================================================================================

struct GPUGaussianSceneCompressed {
	QuantizedGaussianGeometry* __restrict__ d_gaussians = nullptr;
	GAUSSIAN_OPACITY_TYPE* __restrict__ d_opacities = nullptr;
	GAUSSIAN_COLOR_STORAGE_TYPE* __restrict__ d_colors = nullptr;
	CompactSH* __restrict__ d_shs = nullptr;
	GaussianSceneBounds sceneBounds;
	uint32_t numGaussians;

	__host__ __device__ __forceinline__ GPUGaussianSceneCompressed() = default;

	__host__ inline void free() {
		if (d_gaussians) cudaFree(d_gaussians);
		if (d_opacities) cudaFree(d_opacities);
		if (d_colors) cudaFree(d_colors);
		if (d_shs) cudaFree(d_shs);
		d_shs = nullptr;
		d_gaussians = nullptr;
		d_opacities = nullptr;
		d_colors = nullptr;
	}

	__host__ inline void upload_from_cpu(const std::vector<CPUGaussian>& gaussians) {
		free();
		numGaussians = gaussians.size();
		sceneBounds = GaussianSceneBounds(gaussians);

		ERRCHECK(cudaMalloc((void**)&d_gaussians, numGaussians * sizeof(QuantizedGaussianGeometry)));
		ERRCHECK(cudaMalloc((void**)&d_opacities, numGaussians * sizeof(GAUSSIAN_OPACITY_TYPE)));
#ifndef DISABLE_SPHERICAL_HARMONICS
		ERRCHECK(cudaMalloc((void**)&d_shs, numGaussians * sizeof(CompactSH)));
#endif
		ERRCHECK(cudaMalloc((void**)&d_colors, numGaussians * sizeof(GAUSSIAN_COLOR_STORAGE_TYPE)));

		constexpr size_t CHUNK_SIZE = 1000000;
		std::vector<QuantizedGaussianGeometry> gs;
		std::vector<GAUSSIAN_OPACITY_TYPE> opacities;
#ifndef DISABLE_SPHERICAL_HARMONICS
		std::vector<CompactSH> shs;
#endif
		std::vector<GAUSSIAN_COLOR_STORAGE_TYPE> colors;

		gs.reserve(CHUNK_SIZE);
		opacities.reserve(CHUNK_SIZE);
#ifndef DISABLE_SPHERICAL_HARMONICS
		shs.reserve(CHUNK_SIZE);
#endif
		colors.reserve(CHUNK_SIZE);

		size_t chunkStartIdx = 0;

		for (size_t i = 0; i < numGaussians; ++i) {
			const auto& g = gaussians[i];

			opacities.push_back(normalized_float_to_byte(g.opacity));
			GAUSSIAN_COLOR_STORAGE_TYPE sh_eval = encode_color(CudaMath::color_from_base(g.sh));
			colors.emplace_back(sh_eval);
#ifndef DISABLE_SPHERICAL_HARMONICS
			shs.emplace_back(CompactSH(g.sh, sceneBounds.minSH, sceneBounds.maxSH));
#endif
			QuantizedGaussianGeometry cg = encode_gaussian_geometry(g.mean, g.rotation, g.scale, sceneBounds);
			gs.push_back(cg);

			if (gs.size() == CHUNK_SIZE || i == numGaussians - 1) {
				size_t currentChunkSize = gs.size();

				ERRCHECK(cudaMemcpy(d_gaussians + chunkStartIdx, gs.data(), currentChunkSize * sizeof(QuantizedGaussianGeometry), cudaMemcpyHostToDevice));
				ERRCHECK(cudaMemcpy(d_opacities + chunkStartIdx, opacities.data(), currentChunkSize * sizeof(GAUSSIAN_OPACITY_TYPE), cudaMemcpyHostToDevice));
#ifndef DISABLE_SPHERICAL_HARMONICS
				ERRCHECK(cudaMemcpy(d_shs + chunkStartIdx, shs.data(), currentChunkSize * sizeof(CompactSH), cudaMemcpyHostToDevice));
#endif
				ERRCHECK(cudaMemcpy(d_colors + chunkStartIdx, colors.data(), currentChunkSize * sizeof(GAUSSIAN_COLOR_STORAGE_TYPE), cudaMemcpyHostToDevice));

				chunkStartIdx += currentChunkSize;

				gs.clear();
				opacities.clear();
#ifndef DISABLE_SPHERICAL_HARMONICS
				shs.clear();
#endif
				colors.clear();
			}
		}
	}

	__device__ __forceinline__ GaussianGeometry get_gaussian_transform(
		size_t idx) const
	{
		return decode_gaussian_geometry(d_gaussians[idx], sceneBounds);
	}

	__device__ __forceinline__ float get_opacity(size_t idx) const {
		return glm::min(decode_opacity(d_opacities[idx]), MAX_GAUSSIAN_OPACITY);
	}

	__device__ __forceinline__ glm::vec3 get_sh(
		size_t idx,
		int deg,
		const glm::vec3 pos,
		const glm::vec3 campos) const
	{
#ifndef DISABLE_SPHERICAL_HARMONICS
		return CudaMath::computeColorFromCompactSH(
			deg, pos, campos, d_shs[idx], sceneBounds.minSH, sceneBounds.maxSH);
#else
		return decode_color(d_colors[idx]);
#endif
	}

	__device__ __forceinline__ void set_color(size_t idx, const glm::vec3 c) {
		d_colors[idx] = encode_color(c);
	}

	__host__ __device__ __forceinline__ uint64_t get_color(size_t idx) const {
		glm::vec3 c = decode_color(d_colors[idx]);
		uint32_t r12 = (uint32_t)(c.r * 255.0f) & 0xFFF;
		uint32_t g12 = (uint32_t)(c.g * 255.0f) & 0xFFF;
		uint32_t b12 = (uint32_t)(c.b * 255.0f) & 0xFFF;
		uint64_t packed = (uint64_t(r12) << 24) | (uint64_t(g12) << 12) | (uint64_t(b12) << 0);
		return packed;
	}
};


//========================================================================================================
// Full geometry Scene
//========================================================================================================

struct GPUGaussianSceneUncompressed {
	GaussianGeometry* __restrict__ d_gaussians = nullptr;
	float* __restrict__ d_opacities = nullptr;
	GAUSSIAN_COLOR_STORAGE_TYPE* __restrict__ d_colors = nullptr;
	float* __restrict__ d_shs = nullptr;
	uint32_t numGaussians;

	__host__ __device__ __forceinline__ GPUGaussianSceneUncompressed() = default;

	__host__ inline void free() {
		if (d_gaussians) cudaFree(d_gaussians);
		if (d_opacities) cudaFree(d_opacities);
		if (d_colors) cudaFree(d_colors);
		if (d_shs) cudaFree(d_shs);
		d_shs = nullptr;
		d_gaussians = nullptr;
		d_opacities = nullptr;
		d_colors = nullptr;
	}

	__host__ inline void upload_from_cpu(const std::vector<CPUGaussian>& gaussians) {
		free();
		numGaussians = gaussians.size();

		std::vector<GaussianGeometry> gs;
		std::vector<float> opacities;
		std::vector<GAUSSIAN_COLOR_STORAGE_TYPE> colors;
		gs.reserve(numGaussians);
		opacities.reserve(numGaussians);
		colors.reserve(numGaussians);

#ifndef DISABLE_SPHERICAL_HARMONICS
		std::vector<float> shs;
		shs.reserve(numGaussians * 48);
#endif

		for (const auto& g : gaussians) {
			gs.push_back({ g.mean, g.rotation, g.scale });

#ifndef DISABLE_SPHERICAL_HARMONICS
			shs.insert(shs.end(), g.sh, g.sh + 48);
#endif

			opacities.push_back(g.opacity);
			GAUSSIAN_COLOR_STORAGE_TYPE sh_eval = encode_color(CudaMath::color_from_base(g.sh));
			colors.push_back(sh_eval);
		}

		ERRCHECK(cudaMalloc((void**) &d_gaussians, numGaussians * sizeof(GaussianGeometry)));
		ERRCHECK(cudaMemcpy(d_gaussians, gs.data(), numGaussians * sizeof(GaussianGeometry), cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**) &d_opacities, numGaussians * sizeof(float)));
		ERRCHECK(cudaMemcpy(d_opacities, opacities.data(), numGaussians * sizeof(float), cudaMemcpyHostToDevice));

#ifndef DISABLE_SPHERICAL_HARMONICS
		ERRCHECK(cudaMalloc((void**) &d_shs, numGaussians * sizeof(float) * 48));
		ERRCHECK(cudaMemcpy(d_shs, shs.data(), numGaussians * sizeof(float) * 48, cudaMemcpyHostToDevice));
#endif
		ERRCHECK(cudaMalloc((void**) &d_colors, numGaussians * sizeof(GAUSSIAN_COLOR_STORAGE_TYPE)));
		ERRCHECK(cudaMemcpy(d_colors, colors.data(), numGaussians * sizeof(GAUSSIAN_COLOR_STORAGE_TYPE), cudaMemcpyHostToDevice));
	}

	__device__ __forceinline__ GaussianGeometry get_gaussian_transform(
		size_t idx) const
	{
		return d_gaussians[idx];
	}

	__device__ __forceinline__ float get_opacity(size_t idx) const {
		return glm::min(decode_opacity(d_opacities[idx]), MAX_GAUSSIAN_OPACITY);
	}

	__device__ __forceinline__ glm::vec3 get_sh(
		size_t idx,
		int deg,
		const glm::vec3 pos,
		const glm::vec3 campos) const
	{
#ifndef DISABLE_SPHERICAL_HARMONICS
		const float* shBlock = d_shs + idx * 48;
		return CudaMath::computeColorFromSHFloat(deg, pos, campos, shBlock);
#else
		return decode_color(d_colors[idx]);
#endif
	}

	__host__ __device__ __forceinline__ void set_color(size_t idx, glm::vec3 color) const {
		d_colors[idx] = encode_color(color);
	}

	__host__ __device__ __forceinline__ uint64_t get_color(size_t idx) const {
		glm::vec3 c = decode_color(d_colors[idx]);
		uint32_t r12 = (uint32_t)(c.r * 255.0f) & 0xFFF;
		uint32_t g12 = (uint32_t)(c.g * 255.0f) & 0xFFF;
		uint32_t b12 = (uint32_t)(c.b * 255.0f) & 0xFFF;
		uint64_t packed = (uint64_t(r12) << 24) | (uint64_t(g12) << 12) | (uint64_t(b12) << 0);
		return packed;
	}
};


#ifdef USE_COMPRESSED_GAUSSIANS
using GPUGaussianScene = GPUGaussianSceneCompressed;
#else
using GPUGaussianScene = GPUGaussianSceneUncompressed;
#endif
