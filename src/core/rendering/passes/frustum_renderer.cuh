#pragma once
#include "core/rendering/render_pass.cuh"
#include "core/rendering/occlusion/depth_mip_chain.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

class FrustumRenderPass: public RenderPass {
public:
    FrustumRenderPass();
    ~FrustumRenderPass();

    void render(glm::vec3* d_image, float* d_depth, const RenderSettings& settings) override;


private:
    glm::vec3 color;
    float thickness;
};

