#include "input_manager.h"
#include <iostream>

glm::vec3 InputManager::movementVector(0.0f);
glm::vec2 InputManager::mouseDelta(0.0f);
int InputManager::mouseScrollDelta(0);
bool InputManager::keys[GLFW_KEY_LAST + 1] = { false };

void InputManager::update() {
    movementVector = glm::vec3(0.0f);

    if (keys[GLFW_KEY_W]) movementVector.z -= 1.0f;
    if (keys[GLFW_KEY_S]) movementVector.z += 1.0f;
    if (keys[GLFW_KEY_A]) movementVector.x -= 1.0f;
    if (keys[GLFW_KEY_D]) movementVector.x += 1.0f;
    if (keys[GLFW_KEY_SPACE]) movementVector.y -= 1.0f;
    if (keys[GLFW_KEY_LEFT_CONTROL] || keys[GLFW_KEY_RIGHT_CONTROL]) movementVector.y += 1.0f;
}

void InputManager::release() {
    std::fill(std::begin(keys), std::end(keys), false);
    movementVector = glm::vec3(0.0f);
    mouseDelta = glm::vec2(0.0f);
    mouseScrollDelta = 0;
}

glm::vec3 InputManager::getMovementVector() {
    return movementVector;
}

glm::vec2 InputManager::getMouseDelta() {
    return mouseDelta;
}

bool InputManager::isBackSpacePressed()
{
    return keys[GLFW_KEY_F1];
}

bool InputManager::isScreenShotButtonPressed() {
    return keys[GLFW_KEY_F5];
}

int InputManager::getMouseScrollDelta()
{
    return mouseScrollDelta;
}

void InputManager::onScroll(GLFWwindow* window, double xoffset, double yoffset)
{
    //std::cout << yoffset << std::endl;
    mouseScrollDelta += yoffset;
}

void InputManager::processInput(GLFWwindow* window) {
    // Process keyboard input
    for (int i = 0; i <= GLFW_KEY_LAST; ++i) {
        keys[i] = glfwGetKey(window, i) == GLFW_PRESS;
    }

    // Process mouse input
    double xpos, ypos;
    static double lastX = 0.0, lastY = 0.0;

    glfwGetCursorPos(window, &xpos, &ypos);
    
    mouseDelta = glm::vec2(xpos - lastX, lastY - ypos);
    lastX = xpos;
    lastY = ypos;
}
