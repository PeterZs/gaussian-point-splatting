#pragma once
#include "transform.h"

class Node {
public:
    Node();
    Transform& getTransform();
    const Transform& getTransform() const;

private:
    Transform transform;
};
