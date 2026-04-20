#include "window.h"
#include <iostream>
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "input_manager.h"
bool Window::mouseLocked = true;

Window::Window(int width, int height) : width(width), height(height), quadRenderer(nullptr) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, "Stochastic Gaussian Splatting", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
        });

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }


    // Set V-Sync according to the vsyncEnabled variable
    glfwSwapInterval(vsyncEnabled ? 1 : 0); // 0 disables V-Sync, 1 enables it

    toggleMouseLock(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, InputManager::onScroll);

    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Error: Framebuffer is not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    quadRenderer = new QuadRenderer("assets/shaders/quad_vertex.glsl", "assets/shaders/quad_fragment.glsl");
}

Window::~Window() {
    delete quadRenderer;
    glDeleteTextures(1, &texture);
    glDeleteFramebuffers(1, &framebuffer);
    glfwDestroyWindow(window);
    glfwTerminate();
}


void Window::drawPixelBuffer(const unsigned char* h_image) {
    // Bind the framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    // Bind the texture
    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, h_image);

    // Render the quad
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    quadRenderer->renderQuad(texture);
}

void Window::drawPixelBufferFromPBO(const GLuint pbo) {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
        GL_RGB, GL_UNSIGNED_BYTE, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    quadRenderer->renderQuad(texture);
}


void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        toggleMouseLock(window);
    }

    //if (glfwGetKey(window, GLFW_KEY_F2) == GLFW_PRESS) {
    //    std::cout << "save" << std::endl;
    //    CameraLoader::saveActiveCameraState();
    //}

    //if (glfwGetKey(window, GLFW_KEY_F3) == GLFW_PRESS) {
    //    std::cout << "load" << std::endl;
    //    CameraLoader::loadSystemState();
    //}

    //if ((mods & GLFW_MOD_ALT) && action == GLFW_PRESS)
    //{
    //    if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9)
    //    {
    //        int slot = key - GLFW_KEY_0;  // Convert key code into the number 0..9
    //        std::cout << "Hotkey: Loading scene for slot " << slot << std::endl;
    //        CameraLoaders::renderSettingsState.selected_save_slot = slot;
    //        CameraLoader::loadActiveCameraState();
    //    }
    //}
}

void Window::toggleMouseLock(GLFWwindow* window) {
    mouseLocked = !mouseLocked;
    glfwSetInputMode(window, GLFW_CURSOR, mouseLocked ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}

void Window::processEvents() {
    glfwPollEvents();

    InputManager::processInput(window);
}

GLFWwindow* Window::getGLFWWindow() const {
    return window;
}

unsigned int Window::getTexture() const {
    return texture;
}

bool Window::isMouseLocked() const
{
    return mouseLocked;
}
