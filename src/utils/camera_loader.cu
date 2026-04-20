#include "camera_loader.cuh"
#include <fstream>
#include <iostream>
#include "utils/io.h"
#include "render_settings.cuh"

namespace glm {
    void to_json(nlohmann::json& j, const glm::vec3& v) {
        j = nlohmann::json{ v.x, v.y, v.z };
    }

    void to_json(nlohmann::json& j, const glm::mat3& m)
    {
        j = nlohmann::json{
    { m[0][0], m[0][1], m[0][2] },
    { m[1][0], m[1][1], m[1][2] },
    { m[2][0], m[2][1], m[2][2] }
        };
    }

    void from_json(const nlohmann::json& j, glm::vec3& v) {
        j.at(0).get_to(v.x);
        j.at(1).get_to(v.y);
        j.at(2).get_to(v.z);
    }
    void from_json(const nlohmann::json& j, glm::mat3& m)
    {
        const auto& row0 = j.at(0);
        const auto& row1 = j.at(1);
        const auto& row2 = j.at(2);

        m[0][0] = row0.at(0).get<float>();
        m[0][1] = row0.at(1).get<float>();
        m[0][2] = row0.at(2).get<float>();
        m[1][0] = row1.at(0).get<float>();
        m[1][1] = row1.at(1).get<float>();
        m[1][2] = row1.at(2).get<float>();
        m[2][0] = row2.at(0).get<float>();
        m[2][1] = row2.at(1).get<float>();
        m[2][2] = row2.at(2).get<float>();
    }
}

namespace CameraLoader {
    CameraState toCameraState(const RenderSettings& render_settings, const Camera& camera) {
        glm::mat3 rot_mat = glm::transpose(camera.getTransform().getBasis());
        float fy = Camera::fovdeg2focal(render_settings.fovy, render_settings.resolution.y);
        float fx = fy * render_settings.get_aspectXY();        
        //std::vector<glm::vec3> basis = { rot_mat[0], rot_mat[1], rot_mat[2] };
        return {
            camera.image_id,
            camera.image_name,
            render_settings.resolution.x,
            render_settings.resolution.y,
            camera.getTransform().getPosition(),
            rot_mat,
            fy,
            fx
        };
    }

    void fromCameraState(const bool interpolate, const CameraState& state, RenderSettings& render_settings, Camera& camera) {
        glm::mat3 basis = glm::transpose(state.rotation);
        camera.image_id = state.id;
        camera.image_name = state.img_name;
        if (interpolate) {
            camera.interpolate_to(state.position, basis);
        }
        else {
            camera.getTransform().setPosition(state.position);
            camera.setRotationFromBasis(basis);
        }
        if (!render_settings.override_camera_settings) {
            if (render_settings.can_override_resolution) {
                if (render_settings.load_cam_resolution_height_only)
                {
                    float aspectYX = float(state.height) / state.width;
                    render_settings.resolution = { roundf(render_settings.resolution.x), roundf(render_settings.resolution.x * aspectYX) };
                }
                else {
                    render_settings.resolution = { state.width, state.height };
                }
                    
                render_settings.can_override_resolution = false;
            }
                
            render_settings.fovy = Camera::focal2fovdeg(state.fy, state.height);
        }

    }

    void startCameraPath(int n, std::vector<CameraState>& camera_states, RenderSettings& render_settings, Camera &camera) {
        const int N = camera_states.size();
        if (n <= 0) {
            n = N;
        }
        n = glm::clamp(n, 0, N);
        std::vector<glm::vec3> positions{};
        std::vector<glm::mat3> bases{};

        for (size_t i = 0; i < n; i++)
        {
            positions.push_back(camera_states[i].position);
            bases.push_back(glm::transpose(camera_states[i].rotation));
        }
        fromCameraState(false, camera_states[0], render_settings, camera);
        float frameTime = 1.0f / 24.0f;
        camera.interpolate_list(positions, bases, frameTime);
    }

    void saveCameras(std::vector<CameraState> & camera_states, std::string path) {
        nlohmann::json jsonCameras = camera_states;
        std::ofstream camOut(path);
        if (camOut.is_open()) {
            camOut << jsonCameras.dump(4);
            camOut.close();
        }
        else {
            std::cerr << "Error: Unable to open file for writing: " << path << std::endl;
        }
    }

    bool loadCameras(std::vector<CameraState>& camera_states, std::string path) {
        bool success = true;
        std::ifstream camIn(path);
        if (camIn.is_open()) {
            try {
                nlohmann::json jsonCameras;
                camIn >> jsonCameras;
                camera_states = jsonCameras.get<std::vector<CameraState>>();
            }
            catch (const std::exception& e) {
                std::cerr << "Error reading camera_states file: " << e.what() << std::endl;
                success = false;
            }
            camIn.close();
        }
        else {
            std::cerr << "Error: Unable to open file for reading: " << path << std::endl;
            success = false;
        }

        std::sort(camera_states.begin(), camera_states.end(),
            [](const CameraState& a, const CameraState& b) -> bool {
                return a.id < b.id;
            });
        

        return success;
    }
}
