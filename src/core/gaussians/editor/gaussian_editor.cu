#include "gaussian_editor.cuh"
#include <core/rendering/occlusion/gaussian_bvh.cuh>

void GaussianEditor::LoadFile(std::string file)
{
	//use_full_gaussians = false;
    if (!gaussians.empty()) {
        std::vector<CPUGaussian> scene_gaussians;
        if (GaussianLoader::loadPlyFile(file, scene_gaussians, comments, false)) {
            gaussians.insert(gaussians.end(), scene_gaussians.begin(), scene_gaussians.end());
        }
    }
    else {
        GaussianLoader::loadPlyFile(file, gaussians, comments, false);
    }
}

void GaussianEditor::SaveFile(std::string file)
{
	GaussianLoader::writePlyFile(file, gaussians, comments);
}

void GaussianEditor::compute_bounds(glm::vec3& minCorner, glm::vec3& maxCorner) const {
    minCorner = glm::vec3(std::numeric_limits<float>::max());
    maxCorner = glm::vec3(std::numeric_limits<float>::lowest());
    

    for (size_t i = 0; i < gaussians.size(); i++) {
        maxCorner = glm::max(maxCorner, gaussians[i].mean);
        minCorner = glm::min(minCorner, gaussians[i].mean);
    }
    constexpr float SAFETY_MARGIN = 0.001f;
    maxCorner += SAFETY_MARGIN;
    minCorner -= SAFETY_MARGIN;
}

void GaussianEditor::Repeat(glm::ivec3 repeat) {
    // Compute bounds of current scene

    glm::vec3 minCorner, maxCorner;
    compute_bounds(minCorner, maxCorner);


    glm::vec3 bounds = maxCorner - minCorner;
    glm::vec3 centerOffset = (bounds * glm::vec3(repeat)) * 0.5f - bounds * 0.5f;

    std::vector<CPUGaussian> newGaussians;
    newGaussians.reserve(gaussians.size() * repeat.x * repeat.y * repeat.z);


    for (int x = 0; x < repeat.x; ++x) {
        for (int y = 0; y < repeat.y; ++y) {
            for (int z = 0; z < repeat.z; ++z) {
                glm::vec3 offset = glm::vec3(x, y, z) * bounds - centerOffset;
                for (const auto& gaussian : gaussians) {
                    CPUGaussian newGaussian = gaussian;
                    newGaussian.mean += offset;
                    newGaussians.push_back(newGaussian);
                }
            }
        }
    }

    gaussians = std::move(newGaussians);
}


void GaussianEditor::generate_gaussian_wall() {
    glm::vec3 minCorner, maxCorner;
    compute_bounds(minCorner, maxCorner);
    glm::vec3 extent = maxCorner - minCorner;

    glm::vec3 gaussianScale(0.5f);   // size of each Gaussian
    float spacing = 1.0f;           // grid spacing
    float wall_distance = 2.0f;             // distance outside the bounding box
    uint32_t color = 0xFFFFFFFF;     // white
    glm::vec4 rotation = glm::vec4(1, 0, 0, 0); // identity

    glm::vec3 step = extent.x > extent.z ? glm::vec3(spacing, 0, 0) : glm::vec3(0, 0, spacing);
    glm::vec3 offset = extent.x > extent.z ? glm::vec3(0,0, maxCorner.x + wall_distance) : glm::vec3(maxCorner.z + wall_distance, 0, 0);

    int countY = static_cast<int>(extent.y / spacing);
    int countH = static_cast<int>((extent.x > extent.z ? extent.x : extent.z) / spacing);

    for (int iy = 0; iy <= countY; ++iy) {
        for (int ih = 0; ih <= countH; ++ih) {
            glm::vec3 mean = minCorner + offset + step * static_cast<float>(ih) + glm::vec3(0, iy * spacing, 0);
            gaussians.push_back(CreateGaussian(mean, gaussianScale, rotation, color));
        }
    }
}

void GaussianEditor::sort_bvh_order()
{
    GaussianBVH bvh{};
    bvh.build_bvh_and_reorder(gaussians);
}
