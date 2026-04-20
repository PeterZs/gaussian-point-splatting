#pragma once

#include <cuda_runtime.h>
#include <numeric>
//#include <math_constants.h>

#define CUDA_VERSION 12600
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>


#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#undef CUDA_VERSION

#include "config.h"

namespace CudaMath {
	// Produces a screen-space bounding box for a truncated Gaussian.
	// \param out_aabb_min The minimal x- and y-coordinates of the AABB where
	//        viewport coordinates range from 0 to 1.
	// \param out_aabb_max The maximal x- and y-coordinates of the AABB.
	// \param world_to_projection_space The transform from world to projection
	//        space.
	// \param covariance Symmetric, positive covariance matrix of the Gaussian
	// \param mean The world-space mean of the Gaussian
	// \param truncation At how many multiples of the standard deviation the
	//        Gaussian will be truncated.
	__device__ __forceinline__ void gaussian_aabb(glm::vec2& out_aabb_min, glm::vec2& out_aabb_max,
		const glm::mat4 world_to_projection_space, const glm::mat3 covariance, const glm::vec3 mean, const float truncation) {
		// These are the left and right and top and bottom clipping planes in
		// world space
		glm::vec4 planes[2 * 2];
		planes[0] = glm::vec4(1.0, 0.0, 0.0, 1.0) * world_to_projection_space; // x + w = 0
		planes[1] = glm::vec4(1.0, 0.0, 0.0, -1.0) * world_to_projection_space; // x - w = 0
		planes[2] = glm::vec4(0.0, 1.0, 0.0, 1.0) * world_to_projection_space; // y + w = 0
		planes[3] = glm::vec4(0.0, 1.0, 0.0, -1.0) * world_to_projection_space; // y - w = 0
		// Transform each of these planes to a coordinate frame where mean is
		// in the origin
		for (int i = 0; i != 4; ++i)
			planes[i].w += glm::dot(glm::vec3(planes[i]), mean);
		// Compute the AABB
		glm::vec2 bounds[2];
		float t2 = truncation * truncation;
		for (int i = 0; i != 2; ++i) {
			glm::vec4 g = planes[2 * i + 0];
			glm::vec4 h = planes[2 * i + 1];
			// Solve the following equation for x where p = g + x*(h-g):
			// t2 * dot(p.xyz, covariance * p.xyz) - p.w * p.w = 0
			glm::vec4 d = h - g;
			float c = t2 * glm::dot(glm::vec3(g), covariance * glm::vec3(g)) - g.w * g.w;
			float b = t2 * glm::dot(glm::vec3(g), covariance * glm::vec3(d)) - g.w * d.w;
			float a = t2 * glm::dot(glm::vec3(d), covariance * glm::vec3(d)) - d.w * d.w;
			float discriminant = b * b - a * c;
			float sd = sqrt(discriminant);
			bounds[i] = glm::vec2(-b - sd, -b + sd) / a;
		}
		out_aabb_min.x = glm::min(bounds[0].x, bounds[0].y);
		out_aabb_max.x = glm::max(bounds[0].x, bounds[0].y);
		out_aabb_min.y = glm::min(bounds[1].x, bounds[1].y);
		out_aabb_max.y = glm::max(bounds[1].x, bounds[1].y);
	}

	// Returns the minimal depth in the support of a truncated trivariate Gaussian.
	// \param covariance Symmetric, positive covariance matrix of the Gaussian
	// \param mean The world-space mean of the Gaussian
	// \param truncation At how many multiples of the standard deviation the
	//        Gaussian will be truncated.
	// \param camera_pos The world-space camera position.
	// \param z_axis A vector for the z-axis such that the point
	//        camera_pos + z * z_axis is at depth z.
	// \return The minimal depth (may be negative).
	__device__ __forceinline__ float gaussian_min_depth(const glm::mat3 covariance, glm::vec3 mean, float truncation, glm::vec3 camera_pos, glm::vec3 z_axis) {
		glm::vec3 camera_offset = camera_pos - mean;
		float zo = glm::dot(z_axis, camera_offset);
		float zz = glm::dot(z_axis, z_axis);
		// Assemble coefficients of the quadratic equation
		float c = -truncation * truncation * glm::dot(z_axis, covariance * z_axis) + zo * zo;
		float b = zo * zz;
		float a = zz * zz;
		float discriminant = b * b - a * c;
		return (-b - __fsqrt_rn(fmaxf(0.0f, discriminant))) / a;
	}

	// Given a Gaussian (paramaters as for gaussian_min_depth()) and the homogeneous
	// coordinates of a plane (i.e. dot(vec4(point, 1.0), plane) == 0.0 if point is
	// on the plane), this function returns a positive number if and only if this
	// plane cuts the support of the Gaussian into two parts.
	__device__ __forceinline__ float plane_cuts_gaussian(const glm::mat3 covariance, const glm::vec3 mean, const float truncation, glm::vec4 plane) {
		// Transform the plane to a coordinate frame where mean is in the origin
		plane.w += glm::dot(glm::vec3(plane), mean);
		// Check for the intersection with the quadric
		return truncation * truncation * glm::dot(glm::vec3(plane), covariance * glm::vec3(plane)) - plane.w * plane.w;
	}

	//from: https://github.com/MomentsInGraphics/path_tracer/blob/main/src/shaders/camera_utilities.glsl

	/*! Computes a ray direction for a camera defined using a homogeneous
	transform.
	\param ray_tex_coord The screen-space location of the point in a coordinate
		frame where the left top of the viewport is (0, 0), the right top is
		(1, 0) and the right bottom (1, 1).
	\param world_to_proj_space The world to projection space transform of the
		camera (to be multiplied from the left).
	\return The normalized world-space ray direction for the camera ray.*/
	__device__ __host__ __forceinline__ glm::vec3 get_camera_ray_direction(const glm::vec2 ray_tex_coord, const glm::mat4 world_to_proj_space) {
		glm::vec2 dir_proj = 2.0f * ray_tex_coord - glm::vec2(1.0f);
		glm::vec3 ray_dir;
		// This implementation is a bit fancy. The code is automatically generated
		// but the derivation goes as follows: Construct Pluecker coordinates of
		// the ray in projection space, transform those to world space, then take
		// the intersection with the plane at infinity. Of course, the parts that
		// only operate on world_to_proj_space could be pre-computed, but 24 fused
		// multiply-adds is not so bad and I like how broadly applicable this
		// implementation is.
		ray_dir.x = (world_to_proj_space[1][1] * world_to_proj_space[2][3] - world_to_proj_space[1][3] * world_to_proj_space[2][1]) * dir_proj.x;
		ray_dir.x += (world_to_proj_space[1][3] * world_to_proj_space[2][0] - world_to_proj_space[1][0] * world_to_proj_space[2][3]) * dir_proj.y;
		ray_dir.x += world_to_proj_space[1][0] * world_to_proj_space[2][1] - world_to_proj_space[1][1] * world_to_proj_space[2][0];
		ray_dir.y = (world_to_proj_space[0][3] * world_to_proj_space[2][1] - world_to_proj_space[0][1] * world_to_proj_space[2][3]) * dir_proj.x;
		ray_dir.y += (world_to_proj_space[0][0] * world_to_proj_space[2][3] - world_to_proj_space[0][3] * world_to_proj_space[2][0]) * dir_proj.y;
		ray_dir.y += world_to_proj_space[0][1] * world_to_proj_space[2][0] - world_to_proj_space[0][0] * world_to_proj_space[2][1];
		ray_dir.z = (world_to_proj_space[0][1] * world_to_proj_space[1][3] - world_to_proj_space[0][3] * world_to_proj_space[1][1]) * dir_proj.x;
		ray_dir.z += (world_to_proj_space[0][3] * world_to_proj_space[1][0] - world_to_proj_space[0][0] * world_to_proj_space[1][3]) * dir_proj.y;
		ray_dir.z += world_to_proj_space[0][0] * world_to_proj_space[1][1] - world_to_proj_space[0][1] * world_to_proj_space[1][0];
		return normalize(ray_dir);
	}

	__device__ __forceinline__ glm::vec3 get_camera_ray_origin(glm::vec2 ray_tex_coord, glm::mat4 proj_to_world_space) {
		glm::vec4 pos_0_proj = glm::vec4(2.0f * ray_tex_coord - glm::vec2(1.0f), 1.0f, 1.0f);
		glm::vec4 pos_0_world = proj_to_world_space * pos_0_proj;
		return glm::vec3(pos_0_world) / pos_0_world.w;
	}

	__device__ __host__ __forceinline__ glm::mat3 get_camera_ray_direction_matrix(glm::mat4 world_to_proj_space) {
		// Build the raw cofactor matrix C
		glm::mat3 C;
		C[0][0] = world_to_proj_space[1][1] * world_to_proj_space[2][3] - world_to_proj_space[1][3] * world_to_proj_space[2][1];
		C[0][1] = world_to_proj_space[0][3] * world_to_proj_space[2][1] - world_to_proj_space[0][1] * world_to_proj_space[2][3];
		C[0][2] = world_to_proj_space[0][1] * world_to_proj_space[1][3] - world_to_proj_space[0][3] * world_to_proj_space[1][1];

		C[1][0] = world_to_proj_space[1][3] * world_to_proj_space[2][0] - world_to_proj_space[1][0] * world_to_proj_space[2][3];
		C[1][1] = world_to_proj_space[0][0] * world_to_proj_space[2][3] - world_to_proj_space[0][3] * world_to_proj_space[2][0];
		C[1][2] = world_to_proj_space[0][3] * world_to_proj_space[1][0] - world_to_proj_space[0][0] * world_to_proj_space[1][3];

		C[2][0] = world_to_proj_space[1][0] * world_to_proj_space[2][1] - world_to_proj_space[1][1] * world_to_proj_space[2][0];
		C[2][1] = world_to_proj_space[0][1] * world_to_proj_space[2][0] - world_to_proj_space[0][0] * world_to_proj_space[2][1];
		C[2][2] = world_to_proj_space[0][0] * world_to_proj_space[1][1] - world_to_proj_space[0][1] * world_to_proj_space[1][0];

		return C;
	}



	__device__ __host__ __forceinline__ glm::mat3 get_camera_ray_direction_matrix_full(glm::mat4 world_to_proj_space) {
		// Build the raw cofactor matrix C
		glm::mat3 C = get_camera_ray_direction_matrix(world_to_proj_space);
		// Basis matrix B = [[2,0,-1],[0,2,-1],[0,0,1]]
		glm::mat3 B(2, 0, 0,
			0, 2, 0,
			-1, -1, 1);

		// Multiply explicitly: M = C * B
		return C * B;
	}


	__device__ __forceinline__ float gaussian_min_cos(const glm::vec2 minCorner, const glm::vec2 maxCorner, const glm::vec3 forward, const glm::mat4 world_to_proj_space) {

		return glm::min(glm::dot(get_camera_ray_direction(glm::vec2(minCorner.x, minCorner.y), world_to_proj_space), forward),
			glm::min(glm::dot(get_camera_ray_direction(glm::vec2(minCorner.x, maxCorner.y), world_to_proj_space), forward),
				glm::min(glm::dot(get_camera_ray_direction(glm::vec2(maxCorner.x, minCorner.y), world_to_proj_space), forward),
					glm::dot(get_camera_ray_direction(glm::vec2(maxCorner.x, maxCorner.y), world_to_proj_space), forward))));
	}


	__device__ __forceinline__ bool outside_plane_test(glm::mat3 covariance, glm::vec3 mean, float truncation, glm::vec4 plane) {
		bool mean_outside_plane = glm::dot(glm::vec3(plane), mean) + plane.w < 0.0f;
		bool cuts_plane = plane_cuts_gaussian(covariance, mean, truncation, plane) > 0.0f;
		return mean_outside_plane && !cuts_plane;
	}


	__device__ __forceinline__ bool gaussian_in_frustum(
		const glm::mat3& covariance, const glm::vec3& mean,
		float truncation, const glm::vec4* d_frustumPlanes)
	{
		for (int k = 0; k < 6; ++k) {
			if (outside_plane_test(covariance, mean, truncation, d_frustumPlanes[k]))
				return false;
		}
		return true;
	}

	__device__ __forceinline__ bool is_gaussian_mean_in_clip_box(const glm::mat4& mat)
	{
		// Transform points using the matrix
		float4 tp0 = make_float4(
			mat[3][0],
			mat[3][1],
			0.0f,
			mat[3][3]
		);

		float2 projectedP0 = make_float2(tp0.x / (tp0.w + 1e-5f), tp0.y / (tp0.w + 1e-5f));
		float projZ = mat[3][2] / (tp0.w + 1e-8f);
		//return !(projectedP0.x < -1 || projectedP0.x > 1 || projectedP0.y < -1 || projectedP0.y > 1 || projZ < -1 || projZ > 1);//
		return !(projZ < 0 || projZ > 1);//only check z.
	}

	__device__ __forceinline__ glm::vec3 transformPoint4x3(glm::vec3 point, glm::mat4 mat) {
		const glm::mat3 B(mat);
		glm::vec3 o = { mat[3][0], mat[3][1], mat[3][2] };
		return B * point + o;
	}

	// Forward version of 2D covariance matrix computation
	//cov2d is symmetric, so we only store 3 values in xyz
	// [x y]
	// [y z]
	__device__ __forceinline__ glm::vec3 computeCov2D(const glm::mat4& view_matrix, const glm::mat3& cov3D, const glm::vec3 mean, glm::vec2 tan_fov, glm::vec2 focal)
	{
		// The following models the steps outlined by equations 29
		// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
		// Additionally considers aspect / scaling of viewport.
		// Transposes used to account for row-/column-major conventions.
		const glm::mat3 W(view_matrix);
		glm::vec3 t = transformPoint4x3(mean, view_matrix);

		const float limx = 1.3f * tan_fov.x;
		const float limy = 1.3f * tan_fov.y;
		const float txtz = t.x / t.z;
		const float tytz = t.y / t.z;
		t.x = fminf(limx, fmaxf(-limx, txtz)) * t.z;
		t.y = fminf(limy, fmaxf(-limy, tytz)) * t.z;

		glm::mat3 J = glm::mat3(
			focal.x / t.z, 0.0f, -(focal.x * t.x) / (t.z * t.z),
			0.0f, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
			0.0f, 0.0f, 0.0f);

		//glm::mat3 T = glm::transpose(J) * W;
		glm::mat3 T = glm::transpose(W) * J; //Transpose W to be identical to original implementation

		//glm::mat3 cov = T * cov3D * glm::transpose(T);
		glm::mat3 cov = glm::transpose(T) * cov3D * T;

		return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
	}

	__device__ __forceinline__ bool compute2DGaussianBounds3DGS(
		glm::ivec2& out_min_bounds,
		glm::ivec2& out_max_bounds,
		const glm::ivec2 resolution,
		const glm::vec3 cov2d,
		const glm::vec2 pixel_mean)
	{
		const int TILE_SIZE = 16;

		float a = cov2d.x;
		float c = cov2d.z;

		glm::ivec2 ext_rect = {
			(int)ceilf(GAUSSIAN_TRUNCATION_STD * __fsqrt_rn(a)),
			(int)ceilf(GAUSSIAN_TRUNCATION_STD * __fsqrt_rn(c))
		};

		glm::vec2 min_bounds = pixel_mean - glm::vec2(ext_rect);
		glm::vec2 max_bounds = pixel_mean + glm::vec2(ext_rect);

		int tile_min_x = (int)floorf(min_bounds.x / TILE_SIZE);
		int tile_min_y = (int)floorf(min_bounds.y / TILE_SIZE);
		int tile_max_x = (int)ceilf(max_bounds.x / TILE_SIZE);
		int tile_max_y = (int)ceilf(max_bounds.y / TILE_SIZE);

		int tiles_x = (resolution.x + TILE_SIZE - 1) / TILE_SIZE;
		int tiles_y = (resolution.y + TILE_SIZE - 1) / TILE_SIZE;

		if (tile_max_x <= 0 || tile_max_y <= 0 ||
			tile_min_x >= tiles_x || tile_min_y >= tiles_y)
		{
			return false;
		}

		tile_min_x = fmaxf(tile_min_x, 0);
		tile_min_y = fmaxf(tile_min_y, 0);
		tile_max_x = fminf(tile_max_x, tiles_x);
		tile_max_y = fminf(tile_max_y, tiles_y);

		out_min_bounds = glm::ivec2(tile_min_x * TILE_SIZE,
			tile_min_y * TILE_SIZE);
		out_max_bounds = glm::ivec2(tile_max_x * TILE_SIZE,
			tile_max_y * TILE_SIZE);

		return true;
	}




	__device__ __forceinline__ glm::vec3 conic(const glm::vec3& cov2d, const float inv_det) {
		return glm::vec3{
			cov2d.z * inv_det,
			-cov2d.y * inv_det,
			cov2d.x * inv_det
		};
	}
	//mahalanobis_distance
	__device__  __forceinline__ float gaussian_value(const glm::vec3& conic, const glm::vec2 delta) {
		const float sigma =
			-0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) -
			conic.y * delta.x * delta.y;
		if (sigma > 0.f) {
			return 0.0f;
		}
		return __expf(sigma);
	}

	__forceinline__ __host__ __device__ glm::mat3 quat_to_rotmat(const glm::vec4& quat) {
		return glm::toMat3(glm::quat{ quat.x, quat.y, quat.z, quat.w });
	}


	__forceinline__ __host__ __device__ glm::mat3
		scale_to_mat(const glm::vec3& scale) {
		glm::mat3 S = glm::mat3(1.f);
		S[0][0] = scale.x;
		S[1][1] = scale.y;
		S[2][2] = scale.z;
		return S;
	}

	__device__  __host__ __forceinline__ glm::mat3 scale_rot_to_mat(const glm::vec3& scale, const glm::vec4& quat) {
		glm::mat3 R = quat_to_rotmat(quat);
		glm::mat3 S = scale_to_mat(scale);
		return R * S;
	}

	__device__ __host__ __forceinline__ glm::mat3 scale_rot_to_cov(const glm::vec3& scale, const glm::vec4& quat) {
		glm::mat3 M = scale_rot_to_mat(scale, quat);
		return M * glm::transpose(M);
	}

	__device__ __forceinline__ glm::vec3 orthogonalProjectedCov2D(const glm::mat4& vMatrix, const glm::mat3& cov) {
		glm::mat3 V = glm::mat3(vMatrix);
		glm::mat3 c = V * cov * glm::transpose(V);
		return { float(c[0][0]), float(c[0][1]), float(c[1][1]) };
	}

	__device__ __host__ __forceinline__ glm::mat4 scale_rot_trans_to_mat(const glm::vec3& scale, const glm::vec4& quat, const glm::vec3& translation) {
		//glm::mat3 M = scale_rot_to_cov(scale, quat);
		glm::mat3 R = quat_to_rotmat(quat);
		glm::mat3 S = scale_to_mat(scale);
		glm::mat3 M = R * S;// *glm::transpose(R);
		//return glm::identity<glm::mat4>();
		return
			glm::mat4{
				M[0][0],
				M[0][1],
				M[0][2],
				0.0f,
				M[1][0],
				M[1][1],
				M[1][2],
				0.0f,
				M[2][0],
				M[2][1],
				M[2][2],
				0.0f,
				translation.x,
				translation.y,
				translation.z,
				1.0f
		};
	}

	__device__ __forceinline__ glm::mat4 compute_billboard_model_matrix(const glm::mat4 inv_view_mat, const glm::vec3& translation) {
		glm::mat3 M = glm::mat3(inv_view_mat);

		return
			glm::mat4{
				M[0][0],
				M[0][1],
				M[0][2],
				0.0f,
				M[1][0],
				M[1][1],
				M[1][2],
				0.0f,
				M[2][0],
				M[2][1],
				M[2][2],
				0.0f,
				translation.x,
				translation.y,
				translation.z,
				1.0f
		};
	}

	__forceinline__ __device__ float2 projectToXY(const float4& point) {
		return make_float2(point.x / point.w, point.y / point.w);
	}

	//conversion depth in viewspace
	__forceinline__ __device__ float convert_view_depth(uint64_t depth, float near_plane, float far_plane) {
		return -((static_cast<float>(depth) / DEPTH_MAX) * (far_plane - near_plane) + near_plane);
	}
	__forceinline__ __device__ uint64_t convert_view_depth(float depth, float near_plane, float far_plane) {
		return (static_cast<uint64_t>(glm::clamp((glm::abs(depth) - near_plane) / (far_plane - near_plane), 0.0f, 1.0f) * DEPTH_MAX) & DEPTH_MAX);
	}

	__forceinline__ __device__
		float view_to_projected_depth(float z_view, float near_plane, float far_plane)
	{
		float A = (far_plane + near_plane) / (far_plane - near_plane);
		float B = (2.0f * far_plane * near_plane) / (far_plane - near_plane);

		return A + B / z_view;   // returns NDC depth in [-1, 1]
	}	
}
