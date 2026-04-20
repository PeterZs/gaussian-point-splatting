#pragma once
#include "gaussian_loader.cuh"
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>

namespace GaussianTest {
    void createXYZGizmoScene(std::vector<CPUGaussian>& gaussians);
    float3 normalize(float3 v);
    float3 cross(const float3& a, const float3& b);
    float dot(const float3& a, const float3& b);
    void createSphereScene(std::vector<CPUGaussian>& gaussians, float r);
    void generate();
}
