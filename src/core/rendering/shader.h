#pragma once
#include <glad/glad.h>
#include <string>

class Shader {
public:
    Shader(const char* vertexPath, const char* fragmentPath);
    ~Shader();
    void use() const;
    unsigned int getID() const;

private:
    unsigned int ID;
    void checkCompileErrors(unsigned int shader, std::string type);
};
