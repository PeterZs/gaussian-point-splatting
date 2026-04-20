#pragma once
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

class InputManager {
public:
    static void update();
    static void processInput(GLFWwindow* window);
    static void release();
    static glm::vec3 getMovementVector();
    static glm::vec2 getMouseDelta();
    static bool isBackSpacePressed();
    static bool isScreenShotButtonPressed();
    static int getMouseScrollDelta();
    static void onScroll(GLFWwindow* window, double xoffset, double yoffset);

private:
    static glm::vec3 movementVector;
    static glm::vec2 mouseDelta;
    static int mouseScrollDelta;
    static bool keys[GLFW_KEY_LAST + 1];
};
