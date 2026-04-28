#include <string>
#include <vector>
#include <iostream>
#include <render_settings.cuh>

inline bool parseVec3(int& i, int argc, char** argv, glm::vec3& out) {
    if (i + 3 >= argc) return false;
    out.x = std::stof(argv[++i]);
    out.y = std::stof(argv[++i]);
    out.z = std::stof(argv[++i]);
    return true;
}

inline bool parseVec2(int& i, int argc, char** argv, glm::vec2& out) {
    if (i + 2 >= argc) return false;
    out.x = std::stof(argv[++i]);
    out.y = std::stof(argv[++i]);
    return true;
}

RenderSettings parseCommandLine(int argc, char** argv) {
    RenderSettings settings;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--background") {
            parseVec3(i, argc, argv, settings.backgroundColor);
        }
        else if (arg == "--rotation") {
            parseVec3(i, argc, argv, settings.rotation);
        }
        else if (arg == "--translation") {
            parseVec3(i, argc, argv, settings.translation);
        }
        else if (arg == "--scaling") {
            parseVec3(i, argc, argv, settings.scaling);
        }
        else if (arg == "--fovy") {
            settings.fovy = std::stof(argv[++i]);
            settings.override_camera_settings = true;
        }
        else if (arg == "--samples-per-pixel" || arg == "--samples_per_pixel") {
            settings.render_pass_count = std::stoi(argv[++i]);
        }
        else if (arg == "--max-number-of-points" || arg == "--max_number_of_points") {
            settings.max_number_of_points = std::stoi(argv[++i]);
        }
        else if (arg == "--max-eval-count" || arg == "--max_eval_count") {
            settings.max_eval_count = std::stoi(argv[++i]);
        }
        else if (arg == "--supersampling-factor" || arg == "--supersampling_factor") {
            settings.supersampling_factor = std::stoi(argv[++i]);
        }
        else if (arg == "--resolution") {
            glm::vec2 res{};
            parseVec2(i, argc, argv, res);
            settings.resolution = res;
            settings.override_camera_settings = true;
        }
        else if (arg == "--set-target-width" || arg == "--set_target_width") {
            settings.resolution.x = std::stoi(argv[++i]);
        }
        else if (arg == "--freeze-culling") {
            settings.freeze_culling = true;
        }
        else if (arg == "--enable_profiling") {
            settings.nsight_active = true;
        }
        else if (arg == "--cull-small") {
            settings.cull_small_gaussians = true;
        }
        else if (arg == "--one_point_per_thread" || arg == "--one-point-per-thread") {
            settings.reduce_point_count = false;
        }
        else if (arg == "--disable_hierarchical_culling" || arg == "--disable-hierarchical-culling") {
            settings.enable_hierarchical_culling = false;
        }
        else if (arg == "--disable_occlusion_culling" || arg == "--disable-occlusion-culling") {
            settings.enable_occlusion_culling = false;
        }
        else if (arg == "--disable_gui" || arg == "--disable-gui") {
            settings.render_gui = false;
        }
        else if (arg == "--sort_morton_order" || arg == "--sort-morton-order") {
            settings.sort_morton_order = true;
        }
        else if (arg == "--taa") {
            settings.accumulationMode = AccumulationMode::TAA;
        }
        else if (arg == "--cameras-path" || arg == "--camera-path") {
            std::string path = argv[++i];
            settings.camera_path = path;
        }
        else if (arg == "--play_camera_path" || arg == "--play-camera-path") {
            settings.play_camera_path_on_start = true;
        }
        else if (arg == "--run_tests" || arg == "--run-tests") {
            settings.run_tests = true;
        }
        else if (arg == "--snap_window_to_corner" || arg == "--snap-window-to-corner") {
            settings.snap_window_to_corner = true;
        }
        else if (arg == "--model-path") {
            std::string path = argv[++i];
            settings.model_path = path;
        }
        else if (arg == "--create-evaluation-screenshots" || arg == "--create_evaluation_screenshots") {
            settings.create_evaluation_screenshots = true;
        }
        else {
            std::cout << "Unknown argument: " << arg << "\n";
        }
    }

    return settings;
}
