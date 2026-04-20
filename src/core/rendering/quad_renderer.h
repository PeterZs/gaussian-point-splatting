#pragma once
#include <glad/glad.h>
#include "shader.h"

class QuadRenderer {
public:
    QuadRenderer(const char* vertexPath, const char* fragmentPath);
    ~QuadRenderer();
    void renderQuad(unsigned int texture);

private:
    unsigned int quadVAO, quadVBO;
    Shader shader;
    void initQuad();
};
