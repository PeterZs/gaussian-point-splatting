#include "transform.h"

Transform::Transform()
    : position(0.0f, 0.0f, 0.0f), rotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)), scale(1.0f, 1.0f, 1.0f), dirty(true) {
}

glm::vec3 Transform::getPosition() const { return position; }
glm::quat Transform::getRotation() const { return rotation; }
glm::vec3 Transform::getScale() const { return scale; }

void Transform::setPosition(const glm::vec3& pos) {
    position = pos;
    dirty = true;
}

void Transform::setRotation(const glm::quat& rot) {
    rotation = rot;
    dirty = true;
}

void Transform::setRotationFromBasis(const glm::mat3 basis) {
    rotation = glm::quat_cast(basis);
    dirty = true;
}

glm::mat3 Transform::getBasis() const {
    return glm::toMat4(rotation);
}

void Transform::setScale(const glm::vec3& scl) {
    scale = scl;
    dirty = true;
}

glm::mat4 Transform::getModelMatrix() {
    if (dirty) {
        cachedMatrix = glm::mat4(1.0f);
        cachedMatrix = glm::translate(cachedMatrix, position);
        cachedMatrix *= glm::mat4_cast(rotation);
        cachedMatrix = glm::scale(cachedMatrix, scale);
        dirty = false;
    }
    return cachedMatrix;
}
