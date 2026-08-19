#include "config.h"

#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <cuda_profiler_api.h>
#include "core/rendering/render_server.cuh"
#include "core/window.h"
#include "core/gaussians/editor/gaussian_editor.cuh"
#include "world/camera.h"
#include "core/menu.h"
#include "core/input_manager.h"
#include <iostream>
#include "utils/io.h"
#include "timings/stochastic_timings.cuh"

#include <glm/gtc/matrix_transform.hpp>

#include "core/gaussians/gaussian_test.h"
#include <render_settings.cuh>
#include <parse_args.cuh>
#include <core/random/random.cuh>


bool select_gpu() {
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
		std::cerr << "No CUDA dedvices found!\n";
		return false;
	}

	// Always pick the first device (index 0)
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
		std::cerr << "Failed to get device properties for device 0\n";
		return false;
	}

	if (cudaSetDevice(0) != cudaSuccess) {
		std::cerr << "Failed to set device 0\n";
		return false;
	}

	std::cout << "Selected CUDA device 0: " << prop.name << "\n";
	return true;
}


void render(RenderSettings& settings) {
	glm::ivec2 previous_resolution = settings.resolution;

	std::cout << settings.resolution << std::endl;

	Window window(settings.resolution.x, settings.resolution.y, settings.snap_window_to_corner);
	RenderServer renderServer(settings);

	Menu menu(window.getGLFWWindow());

	Transform sceneTransform;
	const size_t image_size = settings.resolution.x * settings.resolution.y * 3;
	unsigned char* h_image = new unsigned char[image_size];

	std::vector<float> timings{};

	settings.reset_camera = false;

	int frame_index = 0;
	int test_index = 0;

	const std::vector<int> spp_targets = { 1, 4, 16, 64, 256, 1024 };
	int eval_spp_idx = 0;
	bool is_initializing_camera = false;

	if (settings.create_evaluation_screenshots)
		settings.load_camera_state(0);

	while (!glfwWindowShouldClose(window.getGLFWWindow())) {
		if (settings.fully_disable_hierarchical_culling)
			settings.enable_hierarchical_culling = false;
		if (settings.resolution != previous_resolution)
		{
			//settings.resolution = previous_resolution;
			std::cerr << "Resolution change not supported at runtime\n";
		}

		if (settings.nsight_active)
			std::cout << "Rendering frame: " << frame_index << std::endl;
		static auto last = std::chrono::high_resolution_clock::now();
		auto now = std::chrono::high_resolution_clock::now();
		float deltaTime = std::chrono::duration<float>(now - last).count();
		last = now;

		if (settings.create_evaluation_screenshots) {

			int next_spp = renderServer.get_num_progressive_frames() + 1;

			if (is_initializing_camera) {
				if (settings.selectedCamera >= settings.camera_states.size() - 1 || settings.selectedCamera >= settings.max_eval_count)
					return;
				settings.change_camera_index(1);
				eval_spp_idx = 0;
				is_initializing_camera = false;
				settings.reset_accumulation = true;
				next_spp = 1;
			}

			if (eval_spp_idx < spp_targets.size() && next_spp == spp_targets[eval_spp_idx]) {
				std::stringstream ss;
				//ss << "cam" << std::setw(3) << std::setfill('0') << settings.selectedCamera
				ss << settings.camera.image_name
					<< "_spp" << std::setw(4) << std::setfill('0') << next_spp;

				renderServer.set_save_next_frame(true, ss.str());
				eval_spp_idx++;
			}
			if (eval_spp_idx >= spp_targets.size())
				is_initializing_camera = true;
		}


		TimingSystem::StochasticTimings::addTimingEntry();
		TimingSystem::StochasticTimings::getCurrentTiming().FrameTime = deltaTime;

		window.processEvents();
		InputManager::update();

		if (frame_index > 0) {
			if (settings.reset_camera) {
				settings.reset_camera = false;
				settings.camera.setRotationFromBasis(glm::mat3(1.0f));
			}

			bool cameraMoved = settings.camera.update(deltaTime, window.isMouseLocked());
			settings.cameraMoved = cameraMoved || settings.reset_accumulation;
			settings.reset_accumulation = false;
		}

		//update scene transform
		sceneTransform.setPosition(settings.translation);
		sceneTransform.setRotation(glm::radians(settings.rotation));
		sceneTransform.setScale(settings.scaling);
		settings.sceneMatrix = sceneTransform.getModelMatrix();
		settings.freeze_culling = settings.freeze_culling;
		settings.cameraPosition = settings.camera.getTransform().getPosition();

		settings.cameraViewMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f)) * settings.camera.getViewMatrix() * settings.sceneMatrix;
		if (!settings.freeze_culling)
			settings.culledViewMatrix = settings.cameraViewMatrix;
		float aspect = settings.get_aspectXY();

		settings.cameraProjectionMatrix = settings.camera.getProjectionMatrix(settings.fovy, aspect, settings.near_view_plane, settings.far_view_plane);

		settings.cameraNormalProjectionMatrix = settings.camera.getNormalProjectionMatrix(settings.fovy, aspect, settings.near_view_plane, settings.far_view_plane);
		settings.frustumPlanes = settings.camera.getFrustumPlanesFromViewMatrix(settings.fovy, settings.cameraViewMatrix,
			aspect, settings.near_view_plane, settings.far_view_plane);

		settings.frame++;
		settings.copyToGPU();

		static bool has_released_screen_shot_button = true;
		if (InputManager::isScreenShotButtonPressed() && has_released_screen_shot_button) {
			has_released_screen_shot_button = false;
			renderServer.set_save_next_frame(true);
		}
		if (!InputManager::isScreenShotButtonPressed() && !has_released_screen_shot_button) {
			has_released_screen_shot_button = true;
		}

		auto time_before_render = std::chrono::high_resolution_clock::now();
		renderServer.renderFrame(settings);
		auto time_after_render = std::chrono::high_resolution_clock::now();
		float render_delta_time = std::chrono::duration<float>(time_after_render - time_before_render).count();

		settings.previousViewProjectionMatrix = settings.cameraProjectionMatrix * settings.cameraViewMatrix;

		window.drawPixelBufferFromPBO(renderServer.get_pbo());

		static bool has_released_backspace = true;
		if (InputManager::isBackSpacePressed() && has_released_backspace) {
			has_released_backspace = false;
			menu.toggleVisibility();
		}
		if (!InputManager::isBackSpacePressed() && !has_released_backspace) {
			has_released_backspace = true;
		}

		if (!settings.nsight_active && settings.render_gui)
			menu.render(settings);

		glfwSwapBuffers(window.getGLFWWindow());
		InputManager::release();


		if (settings.run_tests) {
			int repeatCount = settings.frames_per_camera_keyframe;
			int numCameras = settings.total_test_keyframes;

			int passesPerCamera = repeatCount + 1;
			settings.change_camera_index(1);

			if (frame_index >= numCameras) {
				timings.push_back(render_delta_time);
			}

			if (frame_index % numCameras == 0 && frame_index > 0) {
				settings.load_camera_state(0);
			}

			if (frame_index >= passesPerCamera * numCameras) {

				std::cout << "---BEGIN_TIMINGS_JSON---\n";

				std::cout << "{\n";
				std::cout << "  \"algorithm\": \"GaussianPointSplatting\",\n";
				std::cout << "  \"model_path\": \"" << settings.model_path << "\",\n";
				std::cout << "  \"timings\": [\n";

				for (int cam = 0; cam < numCameras; ++cam) {
					std::cout << "    { \"camera\": " << cam << ", \"samples\": [";

					for (int rep = 0; rep < repeatCount; ++rep) {
						int idx = cam + rep * numCameras;

						float t_ms = timings[idx] * 1000.0f;
						std::cout << t_ms;
						if (rep + 1 < repeatCount) std::cout << ", ";
					}

					std::cout << "] }";
					if (cam + 1 < numCameras) std::cout << ",";
					std::cout << "\n";
				}

				std::cout << "  ]\n";
				std::cout << "}\n";

				std::cout << "---END_TIMINGS_JSON---\n";

				return;
			}
		}

		//render a couple frames for nsight
		if (settings.nsight_active) {
			if (frame_index == 3)
				return;
		}

		frame_index++;
	}
}


int main(int argc, char** argv) {
	if (!select_gpu())
		return -1;

	RenderSettings settings = parseCommandLine(argc, argv);
	settings.statistics = new Statistics();
	if (!settings.check_ready()) {
		return -1;
	}

	if (!settings.sorted_model_output_path.empty())
	{
		GaussianEditor ge{};
		ge.LoadFile(settings.model_path);
		ge.sort_bvh_order();
		ge.comments.push_back(PlyCommentUtils::makeBVHSorted());
		ge.SaveFile(settings.sorted_model_output_path);
	}

	settings.load_camera();


	if (settings.nsight_active) std::cout << "Detected profiling." << std::endl;
	if (settings.run_tests) {
		std::cout << "Running tests." << std::endl;
		settings.play_camera_path_on_start = false;
	}

	if (settings.play_camera_path_on_start)
		settings.start_camera_path();

	render(settings);

	return 0;
}
