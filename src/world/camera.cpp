#include "camera.h"
#include "core/input_manager.h"
#include <iostream>
#include "config.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <array>

Camera::Camera() : yaw(-90.0f), pitch(0.0f), speed(16.0f), sensitivity(0.05f) {}

void Camera::setRotationFromBasis(const glm::mat3 basis) {

	getTransform().setRotationFromBasis(basis);
}

bool Camera::update(float deltaTime, bool mouseLocked) {
	if (cameraPath != nullptr) {
		auto frame = cameraPath->interpolate(deltaTime);
		getTransform().setRotation(frame.rotation);
		getTransform().setPosition(frame.position);

		//cancel animation if mouse moved
		bool mouseMoved = mouseLocked && (glm::length(InputManager::getMouseDelta()) > 0.1f);

		if (frame.finished || mouseMoved) {
			delete cameraPath;
			cameraPath = nullptr;
		}

	} else {
		glm::vec3 movement = glm::vec3(0.0f);

		//get pitch and yaw from stored quaternion
		float yaw = 0.0f, pitch = 0.0f;

		if (mouseLocked) {
			movement = InputManager::getMovementVector();
			movement = glm::normalize(movement);
			if (glm::any(glm::isnan(movement))) movement = glm::vec3(0.0f);

			//update speed
			speed *= std::pow(2.0f, static_cast<float>(InputManager::getMouseScrollDelta()));
			speed = glm::clamp(speed, 0.5f, 1024.0f);

			movement *= speed * deltaTime;
			glm::vec2 mouseDelta = InputManager::getMouseDelta() * sensitivity;

			yaw += mouseDelta.x;
			pitch += mouseDelta.y;
		}

		// Constrain pitch
		//if (pitch > 89.0f) pitch = 89.0f;
		//if (pitch < -89.0f) pitch = -89.0f;

		glm::quat r = getTransform().getRotation();

		// Apply yaw (global Y-axis)
		glm::quat quatYaw = glm::angleAxis(glm::radians(yaw), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::quat tempOrientation = quatYaw * r;

		// Compute local X-axis after yaw
		glm::vec3 localX = glm::rotate(tempOrientation, glm::vec3(1.0f, 0.0f, 0.0f));

		// Apply pitch (local X-axis)
		glm::quat quatPitch = glm::angleAxis(glm::radians(pitch), localX);
		glm::quat orientation = quatPitch * tempOrientation;

		getTransform().setRotation(orientation);

		glm::vec3 front = glm::normalize(orientation * glm::vec3(0.0f, 0.0f, -1.0f));
		glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
		glm::vec3 up = glm::normalize(glm::cross(right, front));

		getTransform().setPosition(getTransform().getPosition() + front * movement.z + right * movement.x + up * movement.y);
	}

	//check if camera moved
	bool moved = glm::distance(previousPosition, getTransform().getPosition()) > FLT_EPSILON || glm::length(previousRotation - getTransform().getRotation()) > FLT_EPSILON;
	previousPosition = getTransform().getPosition();
	previousRotation = getTransform().getRotation();
	return moved;
}

glm::mat4 Camera::getViewMatrix() {
	return glm::affineInverse(getTransform().getModelMatrix());
}

glm::mat4 Camera::getProjectionMatrix(float fovy, float aspectRatio, float near, float far) const {
	return getNormalProjectionMatrix(fovy, aspectRatio, near, far);
}

glm::mat4 Camera::getNormalProjectionMatrix(float fovy, float aspectRatio, float near, float far) const {
	return glm::perspective(glm::radians(fovy), aspectRatio, near, far);
}

void Camera::interpolate_to(const glm::vec3 target_position, const glm::mat3 target_basis, const float duration)
{
	glm::vec3 startPos = getTransform().getPosition();
	glm::quat startRot = getTransform().getRotation();

	glm::quat targetRot = glm::quat_cast(target_basis);

	cameraPath = new CameraPath(startPos, startRot, target_position, targetRot, duration);
}

void Camera::interpolate_list(const std::vector<glm::vec3> &positions, const std::vector<glm::mat3> &bases, const float time_interval)
{
	if (positions.size() != bases.size() || positions.size() <= 1) {
		std::cout << "Won't handle camera path: less than 2 keyframes" << std::endl;
		return;
	}

	cameraPath = new CameraPath(time_interval);

	for (size_t i = 0; i < positions.size(); i++)
	{
		cameraPath->add_keyframe(positions[i], glm::quat_cast(bases[i]));
	}
}


std::array<glm::vec4, 6> Camera::getFrustumPlanesFromViewMatrix(float fovy,
	const glm::mat4 &view,
	const float aspect,
	const float zNear,
	const float zFar) const
{
	std::array<glm::vec4, 6> planes;

	glm::mat4 proj = getNormalProjectionMatrix(fovy, aspect, zNear, zFar);
	glm::mat4 vp = proj * view;

	glm::mat4 m = glm::transpose(vp);

	// Extract planes
	planes[0] = m[3] + m[0]; // Left
	planes[1] = m[3] - m[0]; // Right
	planes[2] = m[3] + m[1]; // Bottom
	planes[3] = m[3] - m[1]; // Top
	planes[4] = m[3] + m[2]; // Near
	planes[5] = m[3] - m[2]; // Far

	for (auto& p : planes) {
		glm::vec3 n(p);
		float len = glm::length(n);
		if (len > 0.0f) {
			p /= len;
		}
	}

	return planes;
}