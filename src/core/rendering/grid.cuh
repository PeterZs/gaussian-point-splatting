#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <core/gaussians/gaussian_types.cuh>


__device__ __forceinline__ glm::vec3 closest_point_on_line(glm::vec3 line_origin, glm::vec3 line_dir,
    glm::vec3 ray_origin, glm::vec3 ray_dir) {
    glm::vec3 w0 = ray_origin - line_origin;
    float a = dot(ray_dir, ray_dir);
    float b = dot(ray_dir, line_dir);
    float c = dot(line_dir, line_dir);
    float d = dot(ray_dir, w0);
    float e = dot(line_dir, w0);

    float denom = a * c - b * b;
    if (denom == 0) return glm::vec3(FLT_MAX);

    float t = (b * e - c * d) / denom;
    float s = (a * e - b * d) / denom;

    if (t < 0) return glm::vec3(FLT_MAX);

    return line_origin + s * line_dir;
}

__global__ __forceinline__ void draw_grid_kernel(
    glm::vec3* d_out_color,
    int width,
    int height,
    glm::mat4 view_proj,
    glm::mat4 inv_view_proj,
    glm::vec3 camera_position,
    float grid_size = 10.0f,
    float line_width = 0.05f
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Convert pixel coordinates to NDC
    float ndc_x = (2.0f * x) / width - 1.0f;
    float ndc_y = (2.0f * y) / height - 1.0f;

    // Create ray through pixel
    glm::vec4 ray_start_proj = inv_view_proj * glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
    glm::vec4 ray_end = inv_view_proj * glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
    ray_start_proj /= ray_start_proj.w;
    ray_end /= ray_end.w;
    glm::vec3 ray_start = glm::vec3(ray_start_proj);

    glm::vec3 ray_dir = glm::normalize(glm::vec3(ray_end) - ray_start);
    glm::vec3 color = glm::vec3(0.0f);

    // Axis drawing
    const float axis_thickness = line_width * 2.0f;
    glm::vec3 closest_point;

    // X-axis (red)
    closest_point = closest_point_on_line(glm::vec3(0), glm::vec3(1, 0, 0), ray_start, ray_dir);
    if (glm::distance(closest_point, ray_start + ray_dir * glm::dot(closest_point - ray_start, ray_dir)) < axis_thickness) {
        color = glm::vec3(1, 0, 0);
    }

    // Y-axis (green)
    closest_point = closest_point_on_line(glm::vec3(0), glm::vec3(0, 1, 0), ray_start, ray_dir);
    if (glm::distance(closest_point, ray_start + ray_dir * glm::dot(closest_point - ray_start, ray_dir)) < axis_thickness) {
        color = glm::vec3(0, 1, 0);
    }

    // Z-axis (blue)
    closest_point = closest_point_on_line(glm::vec3(0), glm::vec3(0, 0, 1), ray_start, ray_dir);
    if (glm::distance(closest_point, ray_start + ray_dir * glm::dot(closest_point - ray_start, ray_dir)) < axis_thickness) {
        color = glm::vec3(0, 0, 1);
    }

    // XZ grid lines    
    float t = (-ray_start.y) / ray_dir.y;
    if (t > 0 && glm::dot(color, color) < 0.01f) {
        glm::vec3 hit = ray_start + ray_dir * t;
        float rgw = 1.0f;
        float grid_thickness = line_width;

//#pragma unroll
        for (size_t i = 0; i < 2; i++)
        {            
            if ((glm::abs(hit.x) < 2.0f * grid_thickness || glm::abs(hit.z) < 2.0f * grid_thickness))
                continue;
            // Check proximity to grid lines
            float frac_x = glm::fract(hit.x * rgw);
            float frac_z = glm::fract(hit.z * rgw);
            float dist_x = glm::min(frac_x, 1.0f - frac_x);
            float dist_z = glm::min(frac_z, 1.0f - frac_z);

            if (dist_x < grid_thickness || dist_z < grid_thickness) {
                float intensity = 1.0f - glm::smoothstep(0.0f, grid_thickness, glm::pow(glm::min(dist_x, dist_z), 2.0f));

                color = glm::mix(color, glm::vec3(0.8f), intensity);
            }
            rgw *= 0.1f;
            grid_thickness *= 4.0f * 0.1f;
        }

        float fade = 1.0f - glm::smoothstep(5.0f, 50.0f, glm::length(hit - camera_position));
        color *= fade;
    }

    if (glm::dot(color, color) > 0.01f)
        d_out_color[y * width + x] = color;
    //d_out_color[y * width + x] = glm::vec3(1.0,0.0,1.0);
}
