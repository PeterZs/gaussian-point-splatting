#pragma once
//#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include "render_settings.cuh"


class Menu {
public:
    Menu(GLFWwindow* window);
    ~Menu();
    void toggleVisibility();
    void render(RenderSettings& settings);

private:
    GLFWwindow* window;
    bool render_menu = true;

};
