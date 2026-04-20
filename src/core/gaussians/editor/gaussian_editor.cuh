#pragma once
#include "core/gaussians/gaussian_loader.cuh"
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>

class GaussianEditor {
public:
	GaussianEditor() {};
	void LoadFile(std::string file);
	void SaveFile(std::string file);
	void compute_bounds(glm::vec3& minCorner, glm::vec3& maxCorner) const;
	void Repeat(glm::ivec3 repeat);

	void generate_gaussian_wall();
	void sort_bvh_order();

	std::vector<PlyComment> comments;

private:
	std::vector<CPUGaussian> gaussians;	
};
