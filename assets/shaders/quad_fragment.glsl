#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenTexture;

void main() {
    FragColor = texture(screenTexture, TexCoord);
    //vec2 texelSize = 1.0 / textureSize(screenTexture, 0);
    //vec2 centeredUV = TexCoord + texelSize * 0.5;
    //FragColor = vec4(1.0, 0.0, 1.0, 1.0);
    //FragColor = vec4(centered_uv, 1, 1);
    //FragColor = texture(screenTexture, centeredUV);
}