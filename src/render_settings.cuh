#pragma once
#include <glm/glm.hpp> 
#include <glm/gtc/matrix_transform.hpp>
#include <array> 
#include <string>
#include <vector>

#include "config.h"
#include "utils/utils.cuh"
#include <world/camera.h>
#include <utils/camera_loader.cuh>

#include <filesystem> 
#include <regex> 
#include <iostream>

enum class AccumulationMode {
    ProgressiveRendering,
    TAA,
    None
};

struct Statistics {
    float totalWeight = 0.0f;
    int numPoints = 0;
    int numPointsOne = 0;
    int numPointsTwo = 0;

    int numTilesOne = 0;
    int numTilesTwo = 0;

    int numGaussiansOne = 0;
    int numGaussiansTwo = 0;
};

struct RenderSettings {
    Statistics *statistics;
    glm::ivec2 resolution = { 1920, 1080 };

    std::string model_path = "";
    std::string sorted_model_output_path = "";
    std::string camera_path = "";
    std::string model_name = "";

    // -----------------------------
    // Rendering parameters
    // -----------------------------
    AccumulationMode accumulationMode = AccumulationMode::ProgressiveRendering;
    int render_pass_count = 1;
    int supersampling_factor = 2;
    int immediate_splat_threshold = 8;
    int occlusion_culling_min_reduce_count = 0;
    int max_number_of_points = 250000000;
    bool grid_enabled = false;
    bool render_gui = true;
    bool freeze_culling = false;
    bool enable_occlusion_culling = true;
    bool enable_hierarchical_culling = true;
    bool fully_disable_hierarchical_culling = false;
    bool cull_small_gaussians = false;
    bool use_unbiased_2d_splatting = true;
    bool interpolate_camera = false;
    bool reset_camera = false;
    bool reset_accumulation = false;
    bool nsight_active = false;
    bool reduce_point_count = true;
    bool play_camera_path_on_start = false;
    bool run_tests = false;
    bool snap_window_to_corner = false;
    bool sort_morton_order = false;
    bool take_pmf_screenshot = false;
    bool create_evaluation_screenshots = false;
    bool override_camera_settings = false;
    bool can_override_resolution = true;
    bool load_cam_resolution_height_only  = true; //scale it such that it keeps aspect ratio and horizontal pixel count

    int max_eval_count = 40;
    int frames_per_camera_keyframe = 10;
    int total_test_keyframes = 0;

    glm::vec3 backgroundColor{ 0.0f };
    glm::vec3 rotation{ 0.0f };
    glm::vec3 translation{ 0.0f };
    glm::vec3 scaling{ 1.0f };

    float near_view_plane = 0.2f;
    float far_view_plane = 10000.0f;
    float fovy = 45.0f;// 47.136f; //45.0f

    // -----------------------------
    // Camera + matrices
    // -----------------------------
    Camera camera{};
    glm::vec3 cameraPosition{ 0.0f };
    glm::mat4 sceneMatrix = glm::identity<glm::mat4>();
    glm::mat4 cameraViewMatrix = glm::identity<glm::mat4>();
    glm::mat4 culledViewMatrix = glm::identity<glm::mat4>();
    glm::mat4 cameraProjectionMatrix = glm::identity<glm::mat4>();
    glm::mat4 cameraNormalProjectionMatrix = glm::identity<glm::mat4>();
    glm::mat4 previousViewProjectionMatrix = glm::identity<glm::mat4>();

    std::array<glm::vec4, 6> frustumPlanes{};
    glm::vec4* d_frustumPlanes = nullptr;

    int frame = 0;
    bool cameraMoved = false;

    // -----------------------------
    // Camera storage
    // -----------------------------
    std::vector<CameraLoader::CameraState> camera_states{};
    int selectedCamera = 0;

    // -----------------------------
    // GPU sync
    // -----------------------------
    void copyToGPU() {
        if (!d_frustumPlanes) {
            cudaMalloc((void**) &d_frustumPlanes, sizeof(glm::vec4) * frustumPlanes.size());
        }
        cudaMemcpy(d_frustumPlanes, frustumPlanes.data(),
            sizeof(glm::vec4) * frustumPlanes.size(),
            cudaMemcpyHostToDevice);
    }

    float get_aspectXY() const {
        return float(resolution.x) / resolution.y;
    }

    static std::string normalize_slashes(std::string s) {
        for (char& c : s)
            if (c == '\\') c = '/';
        return s;
    }


    bool check_ready()
    {
        namespace fs = std::filesystem;

        // ------------------------------------------------------------
        // 1. MODEL PATH MUST EXIST
        // ------------------------------------------------------------
        if (model_path.empty()) {
            std::cout << "Please pass a valid path to a ply gaussian splatting model using --model-path\n";
            return false;
        }

        fs::path mp(model_path);

        // ------------------------------------------------------------
        // 2. If model_path is a directory, auto-detect:
        //    point_cloud/<alphabetically_last>/point_cloud.ply
        // ------------------------------------------------------------
        if (fs::is_directory(mp))
        {
            model_name = mp.lexically_normal().filename().string();

            fs::path point_cloud_dir = mp / "point_cloud";
            if (!fs::exists(point_cloud_dir)) {
                std::cout << "Model directory exists but has no 'point_cloud' subfolder: "
                    << point_cloud_dir << "\n";
                return false;
            }

            fs::path best_folder;
            bool found = false;

            for (auto& entry : fs::directory_iterator(point_cloud_dir)) {
                if (!entry.is_directory()) continue;

                if (!found || entry.path().filename().string() > best_folder.filename().string()) {
                    best_folder = entry.path();
                    found = true;
                }
            }

            if (!found) {
                std::cout << "No subfolders found inside: " << point_cloud_dir << "\n";
                return false;
            }

            fs::path candidate = best_folder / "point_cloud.ply";
            if (!fs::exists(candidate)) {
                std::cout << "Alphabetically last folder exists but has no point_cloud.ply: "
                    << candidate << "\n";
                return false;
            }

            model_path = candidate.string();
            std::cout << "Auto-selected model: " << model_path << "\n";
        }
        else if (!fs::exists(mp)) {
            std::cout << "Model path does not exist: " << model_path << "\n";
            return false;
        }
        else {
            model_name = mp.stem().string();
        }

        // ------------------------------------------------------------
        // 3. If camera_path is empty, infer it from model directory
        // ------------------------------------------------------------
        if (camera_path.empty())
        {
            // model_path = x/point_cloud/<folder>/point_cloud.ply
            // parent_path() x3 = x/
            fs::path model_dir = fs::path(model_path).parent_path().parent_path().parent_path();
            fs::path inferred = model_dir / "cameras.json";

            if (fs::exists(inferred)) {
                camera_path = inferred.string();
                std::cout << "Auto-selected camera file: " << camera_path << "\n";
            }
            else {
                std::cout << "Warning: No camera_path provided and no cameras.json found at: "
                    << inferred << "\n";
            }
        }

        model_path = normalize_slashes(model_path);
        camera_path = normalize_slashes(camera_path);

        std::cout << "Loaded model: " << model_name << "\n";

        return true;
    }

    bool has_camera_data() const {
        return !camera_path.empty();
    }

    void load_camera_state(int index) {
        if (!has_camera_data() || camera_states.size() <= 0) return;
        index = glm::clamp(index, 0, int(camera_states.size()) - 1);
        selectedCamera = index;
        CameraLoader::fromCameraState(interpolate_camera, camera_states[selectedCamera], *this, camera);
    }

    void change_camera_index(int change) {
        load_camera_state(selectedCamera + change);
    }

    void start_camera_path() {
        CameraLoader::startCameraPath(camera_states.size(), camera_states, *this, camera);
    }

    void load_camera() {
        if (!has_camera_data()) return;
        if (!CameraLoader::loadCameras(camera_states, camera_path)) {
            camera_path = "";
        }
        load_camera_state(0);
        if (total_test_keyframes <= 0)
            total_test_keyframes = camera_states.size();
    }

    void save_camera() {
        if (!has_camera_data()) return;
        auto state = CameraLoader::toCameraState(*this, camera);
        camera_states[selectedCamera].position = state.position;
        camera_states[selectedCamera].rotation = state.rotation;
        CameraLoader::saveCameras(camera_states, camera_path);
    }
};
