#include "node.h"

Node::Node() {}

Transform& Node::getTransform() { return transform; }
const Transform& Node::getTransform() const { return transform; }
