#pragma once
#include "cuda_runtime.h"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <vector>
#include <string>
#include "render_pass.cuh"

class RenderServer {
public:
    RenderServer(const RenderSettings& settings);
    ~RenderServer();
    void renderFrame(RenderSettings& settings);

    GLuint get_pbo() const;
    void get_image_data(const RenderSettings& settings, unsigned char*& outBuffer);
    void set_save_next_frame(bool save, std::string file_name= "");

    void save_image_to_disk(const RenderSettings& settings, unsigned char* image_data, std::string custom_name="");

    int get_num_progressive_frames() const { return numProgressiveFrames; }

private:
    unsigned char* d_image;
    std::vector<RenderPass*> renderPasses;

    std::string next_filename = "";
    bool saveNextFrame = false;
    int numProgressiveFrames = 0;

    glm::vec3* d_colorBuffer[2];
    float* d_depthBuffer[2];

    GLuint pbo;
    cudaGraphicsResource* cudaPBOResource = nullptr;


    void initializeRenderer(const RenderSettings& settings);
    void export_pfm_from_buffer(const glm::vec3* d_accumBuffer, int width, int height, int numFrames, const std::string& outputPath);
    void destroyRenderer();
};
