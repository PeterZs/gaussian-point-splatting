#include "quad_renderer.h"

QuadRenderer::QuadRenderer(const char* vertexPath, const char* fragmentPath) : shader(vertexPath, fragmentPath) {
    initQuad();
}

QuadRenderer::~QuadRenderer() {
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
}

void QuadRenderer::initQuad() {
    float quadVertices[] = {
        // Positions      // TexCoords
        -1.0f,  3.0f,      0.0f, 2.0f,
        -1.0f, -1.0f,      0.0f, 0.0f,
         3.0f, -1.0f,      2.0f, 0.0f,
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);
}

void QuadRenderer::renderQuad(unsigned int texture) {
    shader.use();
    glBindVertexArray(quadVAO);
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}
