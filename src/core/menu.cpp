#include "menu.h"
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <iostream>
#include "config.h"

Menu::Menu(GLFWwindow* window) : window(window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

Menu::~Menu() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}


std::string formatWithSeparators(int value) {
    std::string numStr = std::to_string(value);
    int len = numStr.length();
    int separatorPos = len % 3;
    if (separatorPos == 0) separatorPos = 3;

    std::string result;
    for (int i = 0; i < len; ++i) {
        if (i == separatorPos && i != 0) {
            result += ',';
            separatorPos += 3;
        }
        result += numStr[i];
    }
    return result;
}

void Menu::toggleVisibility() {
    render_menu = !render_menu;
}

void Menu::render(RenderSettings& settings) {
    if (!render_menu) return;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Menu");

    ImGui::Text("FPS: %.1f (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);

    if (ImGui::BeginTabBar("MainTabs")) {

        if (ImGui::BeginTabItem("Statistics")) {
            ImGui::Text("Pointcount First Pass: %s", formatWithSeparators(settings.statistics->numPointsOne).c_str());
            if (settings.enable_occlusion_culling) {
                ImGui::Text("Pointcount Second Pass: %s", formatWithSeparators(settings.statistics->numPointsTwo).c_str());
            }
            //ImGui::Text("Number of Gaussians First Pass: %s", formatWithSeparators(settings.statistics->numGaussiansOne).c_str());
            //ImGui::Text("Number of Gaussians Second Pass: %s", formatWithSeparators(settings.statistics->numGaussiansTwo).c_str());
            //ImGui::Text("Total Weight: %.2f", settings.statistics->totalWeight);
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Settings")) {
            const char* accumulationModes[] = { "ProgressiveRendering", "TAA", "None" };
            int currentMode = static_cast<int>(settings.accumulationMode);
            if (ImGui::Combo("Accumulation Mode", &currentMode, accumulationModes, IM_ARRAYSIZE(accumulationModes))) {
                settings.accumulationMode = static_cast<AccumulationMode>(currentMode);
                settings.reset_accumulation = true;
            }
            if (ImGui::Button("Reset Camera"))
            {
                settings.reset_camera = true;
                settings.reset_accumulation = true;
            }
#ifdef ENABLE_FREEZING_CULLING
            if (ImGui::Checkbox("Freeze Culling Frustum", &settings.freeze_culling)) {
                settings.reset_accumulation = true;
            }
#endif // ENABLE_FREEZING_CULLING
            if (ImGui::Checkbox("Use unbiased 2D splatting", &settings.use_unbiased_2d_splatting)) {
                settings.reset_accumulation = true;
            }
            if (ImGui::Checkbox("Enable Hierarchical Culling", &settings.enable_hierarchical_culling)) {
                settings.reset_accumulation = true;
            }
            if (ImGui::Checkbox("Enable Occlusion Culling", &settings.enable_occlusion_culling)) {
                settings.reset_accumulation = true;
            }
            if (settings.enable_occlusion_culling) {
                if (ImGui::DragInt("Occlusion Culling Min Reduce Count", &settings.occlusion_culling_min_reduce_count, 1.0f, 0, 10)) {
                    settings.reset_accumulation = true;
                }
            }
            if (ImGui::Checkbox("Cull Small Gaussians", &settings.cull_small_gaussians)) {
                settings.reset_accumulation = true;
            }
            if (ImGui::DragInt("Resolution Super Sampling Factor", &settings.resolution_upscaling_factor, 1.0f, 1, 8)) {
                settings.resolution_upscaling_factor = glm::clamp(settings.resolution_upscaling_factor, 1, 8);
                settings.reset_accumulation = true;
            }
            if (ImGui::Checkbox("Reduce Point Count", &settings.reduce_point_count)) {
                settings.reset_accumulation = true;
            }
            ImGui::Checkbox("Interpolate Camera", &settings.interpolate_camera);

            if (ImGui::DragFloat("FOVy", &settings.fovy, 0.1f, 1.0f, 150.0f, "%.3f")) {
                settings.reset_accumulation = true;
            }
            //if (ImGui::DragFloat("Far View Plane", &settings.far_view_plane, 0.1f, 0.0f, 10000.0f, "%.3f")) {
            //    settings.reset_accumulation = true;
            //}
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Scene")) {
            ImGui::Checkbox("Enable Grid", &settings.grid_enabled);
            ImGui::ColorEdit3("Background Color", glm::value_ptr(settings.backgroundColor));
            ImGui::DragFloat3("Rotation", glm::value_ptr(settings.rotation), 1.0f, -180.0f, 180.0f, "%.3f");
            ImGui::DragFloat3("Translation", glm::value_ptr(settings.translation), 1.0f, -100.0f, 100.0f, "%.3f");
            ImGui::DragFloat3("Scaling", glm::value_ptr(settings.scaling), 0.1f, 0.0f, 10.0f, "%.3f");
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Tools")) {
            if (ImGui::Button("Take PMF screenshot")) {
                settings.take_pmf_screenshot = true;
            }
            ImGui::EndTabItem();
        }

        //int selected_slot = CameraLoaders::staticSystemState.selected_save_slot;
        //const char* items[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
        //if (ImGui::Combo("Save Slot", &selected_slot, items, IM_ARRAYSIZE(items))) {
        //    CameraLoaders::staticSystemState.selected_save_slot = selected_slot;
        //    CameraLoader::saveSystemState();
        //}


        //int maxSlot = CameraLoaders::get_camera_state_count();
        //int& selected_slot = CameraLoaders::renderSettingsState.selected_save_slot;
        int maxSlot = settings.camera_states.size();
        int& selected_slot = settings.selectedCamera;
        if (maxSlot > 0) {
            if (ImGui::Button("Play Camera Path")) {
                settings.start_camera_path();
                //CameraLoader::startCameraPath();
            }
            if (ImGui::Button("-")) {
                settings.change_camera_index(-1);
            }
            ImGui::SameLine();
            int tempSlot = selected_slot;
            if (ImGui::InputInt("##SaveSlotInput", &tempSlot, 0, 0)) {
                // This branch runs every frame the value changes (on arrow clicks, typing, etc.)
            }
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                // User finished editing (pressed Enter, tabbed out, or clicked away)
                tempSlot = glm::clamp(tempSlot, 0, maxSlot);
                selected_slot = tempSlot;

                //settings.save_camera();
                settings.load_camera_state(settings.selectedCamera);
            }
            ImGui::SameLine();
            if (ImGui::Button("+")) {
                settings.change_camera_index(1);
            }

            if (ImGui::Button("Save Camera")) {
                settings.save_camera();
            }
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
