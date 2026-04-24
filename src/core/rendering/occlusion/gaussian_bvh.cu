#include "gaussian_bvh.cuh"

#include "utils/utils.cuh"
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#include <queue>
#include <stack>
#include <thrust/execution_policy.h>

#include "config.h"

//#define PRINT_OCCLUSION_COUNT

void GaussianBVH::collect_leaf_order(const tinybvh::BVH& bvh, std::vector<uint32_t>& leafOrder, const int gaussiansCount) {
	std::stack<int> st; st.push(0);
	while (!st.empty()) {
		int id = st.top(); st.pop();

		if (id >= bvh.allocatedNodes) continue;
		const auto& n = bvh.bvhNode[id];
		if (n.isLeaf()) {
			if (n.leftFirst >= gaussiansCount) continue;
			uint32_t base = (uint32_t)n.leftFirst;
			for (size_t i = 0; i < n.triCount; i++)
			{
				leafOrder.push_back(base + i);
			}
		}
		else {
			st.push(n.leftFirst);
			st.push(n.leftFirst + 1);
		}
	}
}

#ifdef REORDER_SIMPLE
void GaussianBVH::reorder(std::vector<CPUGaussian>& gaussians,
	const std::vector<uint32_t>& leafOrder) {
	const size_t N = gaussians.size();
	std::vector<CPUGaussian> tmp(N);
	for (size_t newIdx = 0; newIdx < N; ++newIdx) {
		tmp[newIdx] = std::move(gaussians[leafOrder[newIdx]]);
	}
	gaussians.swap(tmp);
}
#else
void GaussianBVH::reorder(std::vector<CPUGaussian>& gaussians,
	const std::vector<uint32_t>& leafOrder) {
	const size_t N = gaussians.size();
	if (leafOrder.size() != N) return; // sanity check

	// Build inverse permutation: oldIdx -> newIdx
	std::vector<uint32_t> perm(N);
	for (uint32_t newIdx = 0; newIdx < N; ++newIdx) {
		uint32_t oldIdx = leafOrder[newIdx];
		if (oldIdx >= N) return; // invalid input
		perm[oldIdx] = newIdx;
	}

	std::vector<char> visited(N, 0);

	for (size_t i = 0; i < N; ++i) {
		if (visited[i] || perm[i] == i) {
			visited[i] = 1;
			continue;
		}

		size_t j = i;
		CPUGaussian tmp = std::move(gaussians[i]);

		while (!visited[j]) {
			visited[j] = 1;
			size_t k = perm[j];
			if (visited[k]) {
				gaussians[j] = std::move(tmp);
				break;
			}
			else {
				gaussians[j] = std::move(gaussians[k]);
				j = k;
			}
		}
	}
}
#endif

void GaussianBVH::flatten_level(const tinybvh::BVH& bvh,
	int targetDepth,
	std::vector<GaussianBVHNode>& flatNodes) {
	struct Item { uint32_t id; int depth; };
	std::queue<Item> q; q.push({ 0,0 });
	while (!q.empty()) {
		Item item = q.front(); q.pop();
		const auto& n = bvh.bvhNode[item.id];
		if (item.depth == targetDepth) {
			GaussianBVHNode gn;
			gn.aabbMin = glm::vec3(n.aabbMin.x, n.aabbMin.y, n.aabbMin.z);
			gn.aabbMax = glm::vec3(n.aabbMax.x, n.aabbMax.y, n.aabbMax.z);
			gn.leafBegin = n.leftFirst;   // tinybvh stores leaf range after reorder
			gn.leafCount = n.triCount;
			flatNodes.push_back(gn);
		}
		else if (!n.isLeaf()) {
			q.push({ n.leftFirst, item.depth + 1 });
			q.push({ n.leftFirst + 1,item.depth + 1 });
		}
	}
}

// Assumes gaussians are already reordered in-place by leaf order.
void GaussianBVH::build_fixed_blocks(const std::vector<CPUGaussian>& gaussians,
	uint32_t blockSize,
	std::vector<GaussianBVHNode>& outNodes)
{
	const uint32_t N = (uint32_t)gaussians.size();
	if (N == 0) { outNodes.clear(); return; }

	outNodes.clear();
	outNodes.reserve((N + blockSize - 1) / blockSize);

	for (uint32_t begin = 0; begin < N; begin += blockSize) {
		const uint32_t end = std::min(begin + blockSize, N);
		const uint32_t count = end - begin;

		glm::vec3 bmin(std::numeric_limits<float>::max());
		glm::vec3 bmax(-std::numeric_limits<float>::max());

		for (uint32_t i = begin; i < end; ++i) {
			glm::vec3 gmin, gmax;
			glm::mat3 cov = CudaMath::scale_rot_to_cov(gaussians[i].scale, gaussians[i].rotation);
			CudaMath::aabb_from_gaussian(gaussians[i].mean, cov, gmin, gmax, GAUSSIAN_TRUNCATION_STD);

			bmin = glm::min(bmin, gmin);
			bmax = glm::max(bmax, gmax);
		}

		GaussianBVHNode node;
		node.aabbMin = bmin;
		node.aabbMax = bmax;
		node.leafBegin = begin;
		node.leafCount = count;
		outNodes.push_back(node);
	}
}

void GaussianBVH::build_bvh_and_reorder(std::vector<CPUGaussian>& gaussians) {
	verts.clear();
	verts.resize(gaussians.size() * 3);

	for (size_t i = 0; i < gaussians.size(); ++i) {
		glm::vec3 aabb_min, aabb_max;
		glm::mat3 cov = CudaMath::scale_rot_to_cov(gaussians[i].scale, gaussians[i].rotation);
		CudaMath::aabb_from_gaussian(gaussians[i].mean, cov, aabb_min, aabb_max, GAUSSIAN_TRUNCATION_STD);
		gaussian_to_triangle(aabb_min, aabb_max, &verts[i * 3]);
	}

	int N = static_cast<int>(gaussians.size());

	bvh.Build(reinterpret_cast<tinybvh::bvhvec4*>(verts.data()), N);
	verts.clear();
	verts.shrink_to_fit();

	std::vector<uint32_t> leafOrder; leafOrder.reserve(N);
	collect_leaf_order(bvh, leafOrder, N);
	
	tinybvh::BVH empty;
	bvh = std::move(empty);
	reorder(gaussians, leafOrder);
}

void GaussianBVH::build_hierarchical_structure(std::vector<CPUGaussian>& gaussians) {
	int primitives_per_node = glm::max(4.0f, glm::floor(gaussians.size() / static_cast<float>(TARGET_NUM_HIERARCHICAL_NODES)));
	//primitives_per_node = 64;

	std::vector<GaussianBVHNode> nodes;
	build_fixed_blocks(gaussians, primitives_per_node, nodes);

	if (dev.d_nodes) cudaFree(dev.d_nodes);
	dev.nodeCount = (uint32_t)nodes.size();
	ERRCHECK(cudaMalloc((void**) &dev.d_nodes, dev.nodeCount * sizeof(GaussianBVHNode)));
	ERRCHECK(cudaMemcpy(dev.d_nodes, nodes.data(),
		dev.nodeCount * sizeof(GaussianBVHNode),
		cudaMemcpyHostToDevice));

	uint32_t words = (dev.nodeCount + 31u) >> 5;
	dev.culledMaskSize = words;
	ERRCHECK(cudaMalloc((void**) &dev.d_culledMask, words * sizeof(uint32_t)));

	// initialise runtime parameters in the device struct
	dev.gaussians_per_block = static_cast<uint32_t>(primitives_per_node);
	dev.tree_span = dev.gaussians_per_block * 32u; 
	//dev.levels = 1; 

	std::cout << "Number of nodes in tree: " << dev.nodeCount << std::endl;
}

void GaussianBVH::free(){
	if (dev.d_nodes)
		ERRCHECK(cudaFree(dev.d_nodes));
	if(dev.d_culledMask)
		ERRCHECK(cudaFree(dev.d_culledMask));
}

void GaussianBVH::release_bvh_host_data() {
	verts.clear();
	verts.shrink_to_fit();

	tinybvh::BVH empty;
	bvh = std::move(empty);

	std::cout << "BVH and vertex buffer released." << std::endl;
}


__device__ __forceinline__ bool aabb_outside_frustum(
	const glm::vec3 bmin,
	const glm::vec3 bmax,
	const glm::vec4* planes)
{
#pragma unroll
	for (int i = 0; i < 6; ++i) {
		const glm::vec3 n = glm::vec3(planes[i]);
		const float     d = planes[i].w;
		// Positive vertex along plane normal
		glm::vec3 v = glm::vec3(
			n.x >= 0.0f ? bmax.x : bmin.x,
			n.y >= 0.0f ? bmax.y : bmin.y,
			n.z >= 0.0f ? bmax.z : bmin.z
		);
		// Outside if the most positive vertex is behind the plane
		if (glm::dot(n, v) + d < 0.0f) return true;
	}
	return false; // Not outside any plane
}


__device__ __forceinline__ bool aabb_project_bounds(
	const glm::vec3 bmin,
	const glm::vec3 bmax,
	const glm::mat4 vp,
	int width, int height,
	glm::ivec2& outMinPix,
	glm::ivec2& outMaxPix,
	float& outFrontDepth
) {
	glm::vec3 corners[8] = {
		{bmin.x,bmin.y,bmin.z},{bmax.x,bmin.y,bmin.z},
		{bmin.x,bmax.y,bmin.z},{bmax.x,bmax.y,bmin.z},
		{bmin.x,bmin.y,bmax.z},{bmax.x,bmin.y,bmax.z},
		{bmin.x,bmax.y,bmax.z},{bmax.x,bmax.y,bmax.z}
	};

	int minx = width - 1, miny = height - 1, maxx = 0, maxy = 0;
	float minDepth = 1e30f;
	bool anyFront = false;

#pragma unroll
	for (int i = 0; i < 8; ++i) {
		glm::vec4 clip = vp * glm::vec4(corners[i], 1.0f);
		if (clip.w <= 0.0f) continue;
		anyFront = true;

		glm::vec3 ndc = glm::vec3(clip) / clip.w;		

		int px = int((ndc.x * 0.5f + 0.5f) * width);
		int py = int((ndc.y * 0.5f + 0.5f) * height);

		minx = glm::min(minx, px); miny = glm::min(miny, py);
		maxx = glm::max(maxx, px); maxy = glm::max(maxy, py);
		minDepth = fminf(minDepth, ndc.z);
	}

	if (!anyFront) return false;

	minx = glm::max(minx, 0); miny = glm::max(miny, 0);
	maxx = glm::min(maxx, width - 1); maxy = glm::min(maxy, height - 1);
	if (minx > maxx || miny > maxy) return false;



	outMinPix = { minx, miny };
	outMaxPix = { maxx, maxy };
	outFrontDepth = minDepth;
	return true;
}

__global__ void compute_occlusion_mask(
	const GaussianBVH::GaussianBVHDevice dev,
	const glm::vec4* frustumPlanes,
	const glm::mat4 vp,
	const DepthMipChainDevice hiz,
	const bool enable_occlusion_culling
	) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t warp_id = threadIdx.x % 32;
	uint32_t warpBase = (idx & ~31u);
	uint32_t wordIdx = warpBase >> 5;

	if (wordIdx >= dev.culledMaskSize) return;
	bool culled = false;

	if (idx < dev.nodeCount) {	

		const GaussianBVH::GaussianBVHNode n = dev.d_nodes[idx];
		if (aabb_outside_frustum(n.aabbMin, n.aabbMax, frustumPlanes)) {
			culled = true;
		}
		else if (enable_occlusion_culling)
		{
			
			glm::ivec2 minPix, maxPix;
			float frontDepth;
			if (aabb_project_bounds(n.aabbMin, n.aabbMax, vp,
				hiz.width, hiz.height,
				minPix, maxPix, frontDepth)) {
				culled = depth_mip_is_occluded(hiz, frontDepth, minPix, maxPix);
			}
		}
	}

	__syncwarp();

	unsigned int mask = __ballot_sync(0xffffffff, culled);
	if ((threadIdx.x & 31) == 0) {
		uint32_t warpBase = idx & ~31u;
		uint32_t wordIdx = warpBase >> 5;
		dev.d_culledMask[wordIdx] = mask;
	}
}



void GaussianBVH::update_hierarchical_culling(const glm::vec3& camera_position, const glm::vec4 *d_planes, const glm::mat4& vpMatrix, const DepthMipChain& depth_mip_chain, const bool enable_occlusion_culling) {
	if (!dev.d_nodes || !dev.d_culledMask || !d_planes) return;

	DepthMipChainDevice hiz = depth_mip_chain.device_view();	

	int blockSize = suggestBlockSize(compute_occlusion_mask);
	dim3 grid = makeGrid1D(dev.nodeCount, blockSize);
	KERNEL_LAUNCH(compute_occlusion_mask, grid, blockSize,
		dev,
		d_planes,
		vpMatrix,
		hiz,
		enable_occlusion_culling
		);	

#ifdef PRINT_OCCLUSION_COUNT
	auto extract_weight = [=] __host__ __device__(const uint32_t word) {
#pragma unroll
		uint32_t n = 0;
		for (int i = 0; i < 32; ++i)
			n += (word >> i) & 1;
		return n;
	};

	uint32_t culledCount = thrust::reduce(thrust::device,
		thrust::make_transform_iterator(dev.d_culledMask, extract_weight),
		thrust::make_transform_iterator(dev.d_culledMask + dev.culledMaskSize, extract_weight)
	);
	culledCount *= dev.gaussians_per_block;

	std::cout << "[Occlusion] Culled Gaussians: " << culledCount << std::endl;
#endif // PRINT_OCCLUSION_COUNT
}