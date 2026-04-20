#pragma once

#include <nlohmann/json.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <array>
#include <vector>
#include <string>

#include "world/camera.h"

struct RenderSettings;

namespace glm {
    void to_json(nlohmann::json& j, const glm::vec3& v);
    void to_json(nlohmann::json& j, const glm::mat3& m);
    void from_json(const nlohmann::json& j, glm::vec3& v);
    void from_json(const nlohmann::json& j, glm::mat3& m);
}

namespace CameraLoader {

    // File paths for saving the states.
    const std::string RENDER_STATE_SAVE_PATH = "assets/saves/render_state.json";

    struct CameraState {
        int id;
        std::string img_name;
        int width;
        int height;
        glm::vec3 position;
        glm::mat3 rotation;
        float fy;
        float fx;

        NLOHMANN_DEFINE_TYPE_INTRUSIVE(CameraState, id, img_name, position, rotation, width, height, fy, fx)
    };

    CameraState toCameraState(const RenderSettings& render_settings, const Camera& camera);

    void fromCameraState(const bool interpolate, const CameraState& state, RenderSettings& render_settings, Camera& camera);

    void startCameraPath(int n, std::vector<CameraState>& camera_states, RenderSettings& render_settings, Camera& camera);

    void saveCameras(std::vector<CameraLoader::CameraState>& camera_states, std::string path);
    bool loadCameras(std::vector<CameraLoader::CameraState>& camera_states, std::string path);

}

