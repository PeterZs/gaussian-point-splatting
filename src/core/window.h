#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "core/rendering/quad_renderer.h"
#include <world/camera.h>

class Window {
public:
    Window(int width, int height, bool snap_window_to_corner);
    void toggleVSync(bool enable);
    ~Window();
    void drawPixelBuffer(const unsigned char* h_image);
    void drawPixelBufferFromPBO(const GLuint pbo);
    void processEvents();
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void toggleMouseLock(GLFWwindow* window);
    GLFWwindow* getGLFWWindow() const;
    unsigned int getTexture() const;
    bool isMouseLocked() const;

private:
    GLFWwindow* window;
    int width, height;
    static bool mouseLocked;
    bool vsyncEnabled = false;
    unsigned int framebuffer;
    unsigned int texture;
    QuadRenderer* quadRenderer;
};
