#include "frustum_renderer.cuh"

FrustumRenderPass::FrustumRenderPass() {}
FrustumRenderPass::~FrustumRenderPass() {}

__device__ __forceinline__ glm::vec3 project_to_ndc(const glm::vec4& clip) {
    return glm::vec3(clip) / clip.w; // [-1,1]
}

__device__ __forceinline__ glm::vec2 ndc_to_screen(const glm::vec3& ndc, glm::uvec2 res) {
    return glm::vec2((ndc.x * 0.5f + 0.5f) * res.x,
        (ndc.y * 0.5f + 0.5f) * res.y);
}

__device__ __forceinline__ float segment_param(glm::vec2 p, glm::vec2 a, glm::vec2 b) {
    glm::vec2 ab = b - a;
    glm::vec2 ap = p - a;
    float denom = glm::dot(ab, ab);
    if (denom <= 0.0f) return 0.0f;
    return glm::clamp(glm::dot(ap, ab) / denom, 0.0f, 1.0f);
}

__global__ void visualize_frustum_kernel(glm::vec3* d_image, const float* d_depth,
    glm::uvec2 resolution, glm::vec3 frustum_color, float frustum_width,
    glm::mat4 currentVP, glm::mat4 invStoredVP) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (int)resolution.x || y >= (int)resolution.y) return;

    const int out_idx = y * resolution.x + x;
    glm::vec2 pixel = glm::vec2(x, y);

    // Clip-space unit cube corners (frustum in clip space)
    const float DEPTH_VALUE_NEAR = -0.5;
    const float DEPTH_VALUE_FAR = 0.999;
    const glm::vec3 clipCorners[8] = {
        {-1,-1,DEPTH_VALUE_NEAR}, { 1,-1,DEPTH_VALUE_NEAR}, { 1, 1,DEPTH_VALUE_NEAR}, {-1, 1,DEPTH_VALUE_NEAR}, // near
        {-1,-1, DEPTH_VALUE_FAR}, { 1,-1, DEPTH_VALUE_FAR}, { 1, 1, DEPTH_VALUE_FAR}, {-1, 1, DEPTH_VALUE_FAR}  // far
    };

    // Edges over corner indices
    const int edges[12][2] = {
        {0,1},{1,2},{2,3},{3,0}, // near
        {4,5},{5,6},{6,7},{7,4}, // far
        {0,4},{1,5},{2,6},{3,7}  // sides
    };

    // Compute world-space corners by inverting stored VP, then screen-space via current VP
    glm::vec4 world[8];
    glm::vec2 screen[8];
    for (int i = 0; i < 8; ++i) {
        // world = inv(storedVP) * clipCorner
        glm::vec4 wc = invStoredVP * glm::vec4(clipCorners[i], 1.0f);
        world[i] = wc; // affine inverse, no need to renormalize; wc is world point
        // project into current VP for screen placement
        glm::vec3 ndc = project_to_ndc(currentVP * wc);
        if (ndc.z < -1.0f || ndc.z > 1.0f) return;
        screen[i] = ndc_to_screen(ndc, resolution);
    }

    // Find closest edge and its param; compute exact depth at that param
    bool draw = false;
    float lineDepth = 1.0f;

    for (int e = 0; e < 12; ++e) {
        int i0 = edges[e][0], i1 = edges[e][1];
        glm::vec2 a = screen[i0], b = screen[i1];

        // Thickness test
        float t = segment_param(pixel, a, b);
        glm::vec2 closest = a + t * (b - a);
        float dist = glm::length(pixel - closest);
        if (dist < frustum_width) {
            draw = true;

            // Interpolate world point along the edge at exact t
            glm::vec4 wP = (1.0f - t) * world[i0] + t * world[i1];

            // Project into current VP and compute depth in [0,1]
            glm::vec3 ndc = project_to_ndc(currentVP * wP);
            // Reject if behind the near/far (optional)
            if (ndc.z < -1.0f || ndc.z > 1.0f) continue;

            lineDepth = ndc.z * 0.5f + 0.5f; // [0,1]
            float sceneDepth = d_depth[out_idx];

            //// Depth rejection: only draw if the frustum line is closer than the scene
            //if (lineDepth < sceneDepth) {
            //    draw = true;
            //    break; // draw once per pixel if any edge passes
            //}
        }
    }

    if (draw) {
        d_image[out_idx] = frustum_color;
    }
}

void FrustumRenderPass::render(glm::vec3* d_image, float* d_depth, const RenderSettings& settings)
{
    if (!settings.freeze_culling) return;
    dim3 block(16, 16);
    dim3 grid((settings.resolution.x + block.x - 1) / block.x,
        (settings.resolution.y + block.y - 1) / block.y);

    glm::mat4 pMatrix = settings.cameraNormalProjectionMatrix;

    glm::uvec2 resolution = { settings.resolution.x, settings.resolution.y };
    glm::mat4 currentVP = pMatrix * settings.cameraViewMatrix;
    glm::mat4 storedVP = pMatrix * settings.culledViewMatrix;

    // Use affineInverse on host to avoid doing it per-thread
    glm::mat4 invStoredVP = glm::inverse(storedVP);

    visualize_frustum_kernel << <grid, block >> > (
        d_image,
        d_depth,
        resolution,
        glm::vec3(1.0f, 0.0f, 1.0f), // magenta
        1.5f,                        // thickness in pixels
        currentVP,
        invStoredVP
        );
}
