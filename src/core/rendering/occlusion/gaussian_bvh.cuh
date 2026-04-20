#include <core/gaussians/gaussian_types.cuh>

namespace tinybvh {
	     using bvhint2 = glm::ivec2;
	     using bvhint3 = glm::ivec3;
	     using bvhuint2 = glm::uvec2;
	     using bvhuint3 = glm::uvec3;
	     using bvhuint4 = glm::uvec4;
	     using bvhvec2 = glm::vec2;
	     using bvhvec3 = glm::vec3;
	     using bvhvec4 = glm::vec4;
	     using bvhdbl3 = glm::dvec3;
	     using bvhmat4 = glm::mat4x4;
}
#define TINYBVH_USE_CUSTOM_VECTOR_TYPES
#include "tiny_bvh.h"
#include "depth_mip_chain.cuh"
#include <iostream>

inline void gaussian_to_triangle(const glm::vec3& min, const glm::vec3& max,
	glm::vec4* outVerts) {
	// Pick three corners of the AABB
	outVerts[0] = glm::vec4(min.x, min.y, min.z, 0.0f);
	outVerts[1] = glm::vec4(max.x, min.y, min.z, 0.0f);
	outVerts[2] = glm::vec4(min.x, max.y, min.z, 0.0f);
}

class GaussianBVH {
public:

    // Device-friendly node: just AABB and optional leaf range mapping.
    struct GaussianBVHNode {
        glm::vec3 aabbMin;
        glm::vec3 aabbMax;
        uint32_t  leafBegin;  // index into reordered gaussians (inclusive)
        uint32_t  leafCount;  // number of leaves under this node
    };

    struct GaussianBVHDevice {
        GaussianBVHNode* d_nodes = nullptr;
        uint32_t* d_culledMask = nullptr;
        uint32_t culledMaskSize = 0;   // number of entries in the culled mask
        uint32_t nodeCount = 0;   // number of blocks/nodes
        uint32_t gaussians_per_block;   // e.g. 64
        uint32_t tree_span;             // = gaussians_per_block * 32

        __forceinline__ __device__ bool is_culled(uint32_t gaussian_id) const {
            uint32_t treeId = gaussian_id / tree_span;                // which mask word
            uint32_t within = gaussian_id % tree_span;                // offset within that span
            uint32_t blockId = within / gaussians_per_block;           // 0..31
            uint32_t word = d_culledMask[treeId];
            return ((word >> blockId) & 1u) != 0u;
        }

        //__device__ inline bool is_culled(uint32_t idx) const {
        //    const uint32_t word = d_culledMask[idx >> 5];
        //    const uint32_t bit = (word >> (31 - (idx & 31))) & 1u; // LSB-first (ballot layout)
        //    return bit != 0;
        //}
    };

    GaussianBVH() {
    }

    ~GaussianBVH() {
        free();
    }


    void build_bvh_and_reorder(std::vector<CPUGaussian>& gaussians);

    void build_hierarchical_structure(std::vector<CPUGaussian>& gaussians);

    void release_bvh_host_data();

    void update_hierarchical_culling(const glm::vec3& camera_position, 
        const glm::vec4* d_planes,
        const glm::mat4& vpMatrix,
        const DepthMipChain& depth_mip_chain,
        const bool enable_occlusion_culling);

    GaussianBVHDevice device_view() const { return dev; }

private:
    std::vector<glm::vec4> verts{}; // 3 verts per Gaussian
    tinybvh::BVH bvh{};
    GaussianBVHDevice dev{};

    void free();

    void collect_leaf_order(const tinybvh::BVH& bvh, std::vector<uint32_t>& leafOrder, const int gaussiansCount);

    void reorder(std::vector<CPUGaussian>& gaussians, const std::vector<uint32_t>& leafOrder);

    void flatten_level(const tinybvh::BVH& bvh, int targetDepth, std::vector<GaussianBVHNode>& flatNodes);

    void build_fixed_blocks(const std::vector<CPUGaussian>& gaussians, uint32_t blockSize, std::vector<GaussianBVHNode>& outNodes);
};
