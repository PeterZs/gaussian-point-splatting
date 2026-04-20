#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "gaussian_types.cuh"
#include "ply_comments.h"

class GaussianLoader {
public:
    static bool loadPlyFile(const std::string& filename, std::vector<CPUGaussian>& gaussians, std::vector<PlyComment>& comments, const bool sort_morton_order);
    static void printGaussian(const CPUGaussian& gaussian);
    static void writePlyFile(const std::string& filename, const std::vector<CPUGaussian>& gaussians, const std::vector<PlyComment>& comments = {});
};
