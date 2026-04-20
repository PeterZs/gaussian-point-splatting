#pragma once
#include <cuda_runtime.h>
#include <render_settings.cuh>
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


class RenderPass {
public:
    virtual void render(glm::vec3* d_image, float* d_depth, const RenderSettings& renderSettings) = 0;
    RenderPass() = default;
    virtual ~RenderPass() = default;

    bool enabled = true;
};
