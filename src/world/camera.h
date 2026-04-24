#pragma once
#include "node.h"
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <string>

class Camera : public Node {
public:
    Camera();
    ~Camera() {
        delete cameraPath;
    }

    void setRotationFromBasis(const glm::mat3 basis);

    bool update(float deltaTime, bool mouseLocked);
    glm::mat4 getViewMatrix();
    glm::mat4 getProjectionMatrix(float fovy, float aspectRatio, float near, float far) const;
    glm::mat4 getNormalProjectionMatrix(float fovy, float aspectRatio, float, float) const;

    void interpolate_to(const glm::vec3 target_position, const glm::mat3 target_basis, const float duration = 1.0f);

    void interpolate_list(const std::vector<glm::vec3> &positions, const std::vector<glm::mat3> &bases, const float time_interval);

    std::array<glm::vec4, 6> getFrustumPlanesFromViewMatrix(float fovy, const glm::mat4& view, const float aspect, const float zNear, const float zFar) const;

    float yaw = 0.0f;
    float pitch = 0.0f;
    float speed = 1.0f;
    float sensitivity = 1.0f;

    std::string image_name = "0000";
    int image_id = 0;

    static inline float fovdeg2focal(float fovdeg, int pixels) {
        float fovr = fovdeg * 0.01745329;
        return pixels / (2.0 * tanf(fovr / 2.0));
    }

    static inline float focal2fovdeg(float focal, int pixels) {
        float fovr = 2.0 * atanf(0.5f * pixels / focal);
        return fovr * 57.2957795f;
    }    


private:
    struct CameraKeyframe {
        glm::vec3 position{};
        glm::quat rotation{};
        bool finished = false;
    };

    struct CameraPath {
    private:
        std::vector<CameraKeyframe> keyframes{};
        float time_interval = 0.25f;
        float duration = 0.0f;
        float elapsed = 0.0f;

        void update_duration() {
            duration = time_interval * glm::max(static_cast<float>(keyframes.size() - 1), 0.0f);
        }

    public:
        CameraPath(const float time_interval) : time_interval(time_interval) {}

        CameraPath(const glm::vec3 from_pos, const glm::quat from_rot, const glm::vec3 to_pos, const glm::quat to_rot, const float time_interval) : elapsed(0.0f), time_interval(time_interval) {
            keyframes.push_back({ from_pos, from_rot, false });
            keyframes.push_back({ to_pos, to_rot, false });
            update_duration();
        }

        void add_keyframe(const glm::vec3 pos, const glm::quat rot) {
            keyframes.push_back({ pos, rot, false });
            update_duration();
        }

        inline CameraKeyframe interpolate(float deltaTime) {
            elapsed = glm::clamp(elapsed + deltaTime, 0.0f, duration);
            uint32_t i = static_cast<int>(std::floor(elapsed / time_interval));
            CameraKeyframe& a = keyframes[i];
            CameraKeyframe& b = keyframes[i + 1];

            float t = glm::clamp(elapsed / time_interval - static_cast<float>(i), 0.0f, 1.0f);            

            CameraKeyframe result;
            result.position = glm::mix(a.position, b.position, t);
            result.rotation = glm::slerp(a.rotation, b.rotation, t);
            result.finished = elapsed >= duration;
            return result;
        }
    };

    CameraPath* cameraPath = nullptr;
    glm::quat previousRotation;
    glm::vec3 previousPosition;
};
