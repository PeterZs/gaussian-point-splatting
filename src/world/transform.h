#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <array>

class Transform {
public:
    Transform();

    glm::vec3 getPosition() const;
    glm::quat getRotation() const;
    glm::vec3 getScale() const;

    void setPosition(const glm::vec3& pos);
    void setRotation(const glm::quat& rot);
    void setRotationFromBasis(const glm::mat3 basis);
    void setScale(const glm::vec3& scl);

    glm::mat3 getBasis() const;
    glm::mat4 getModelMatrix();

private:
    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 scale;
    glm::mat4 cachedMatrix;
    bool dirty;
};
