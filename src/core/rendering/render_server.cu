#include "render_server.cuh"
#include "grid.cuh"

#include "core/rendering/passes/post_processing_kernels.cuh"
#include "core/rendering/passes/gaussian_point_splatting.cuh"
#include "core/rendering/passes/frustum_renderer.cuh"
#include "utils/utils.cuh"
#include "utils/io.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "config.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <fstream> 
#include <string>
#include <ctime>
#include <iomanip>

RenderServer::RenderServer(const RenderSettings& settings)
	: d_image(nullptr) {
	initializeRenderer(settings);
}

RenderServer::~RenderServer() {
	destroyRenderer();
}

void RenderServer::initializeRenderer(const RenderSettings& settings) {
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, settings.resolution.x * settings.resolution.y * 3 * sizeof(unsigned char), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	ERRCHECK(cudaGraphicsGLRegisterBuffer(&cudaPBOResource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	ERRCHECK(cudaMalloc(&d_depthBuffer[0], settings.resolution.x * settings.resolution.y * sizeof(float)));
	ERRCHECK(cudaMalloc(&d_depthBuffer[1], settings.resolution.x * settings.resolution.y * sizeof(float)));
	ERRCHECK(cudaMalloc(&d_colorBuffer[0], settings.resolution.x * settings.resolution.y * sizeof(glm::vec3)));
	ERRCHECK(cudaMalloc(&d_colorBuffer[1], settings.resolution.x * settings.resolution.y * sizeof(glm::vec3)));


	int render_pass_count = settings.render_pass_count;
	

	auto stochastic_pass = new GaussianPointSplatting(settings);
	renderPasses.push_back(stochastic_pass);

#ifdef ENABLE_FREEZING_CULLING
	renderPasses.push_back(new FrustumRenderPass());
#endif // ENABLE_FREEZING_CULLING    

}

void RenderServer::destroyRenderer() {
	ERRCHECK(cudaFree(d_depthBuffer[0]));
	ERRCHECK(cudaFree(d_depthBuffer[1]));
	ERRCHECK(cudaFree(d_colorBuffer[0]));
	ERRCHECK(cudaFree(d_colorBuffer[1]));

	for (RenderPass* pass : renderPasses) {
		delete pass;
	}
	renderPasses.clear();
}

__global__ void write_constant_rgb(unsigned char* __restrict__ image,
	int width, int height, int numProgressiveFrames) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int count = width * height;
	if (idx >= count) return;

	unsigned char r = (unsigned char)(255);
	unsigned char g = 0;
	unsigned char b = (unsigned char)(255);

	int base = 3 * idx;
	image[base + 0] = r;
	image[base + 1] = g;
	image[base + 2] = b;
}


void RenderServer::renderFrame(RenderSettings& settings) {
	int a = settings.frame % 2;
	int b = 1 - a;

	for (RenderPass* pass : renderPasses) {
		if (!pass || !pass->enabled) continue;
		pass->render(d_colorBuffer[a], d_depthBuffer[b], settings);
	}

	auto vMatrix = settings.cameraViewMatrix * settings.sceneMatrix;
	auto vpMatrix = settings.cameraProjectionMatrix * vMatrix;
	auto invVpMatrix = glm::inverse(vpMatrix);

	if (settings.grid_enabled) {
		float line_width = 1.5f / glm::min(settings.resolution.x, settings.resolution.y);
		dim3 threads(16, 16);
		dim3 blocks((settings.resolution.x + threads.x - 1) / threads.x,
			(settings.resolution.y + threads.y - 1) / threads.y);

		KERNEL_LAUNCH(draw_grid_kernel, blocks, threads,
			d_colorBuffer[a],
			settings.resolution.x,
			settings.resolution.y,
			vpMatrix,
			invVpMatrix,
			settings.cameraPosition,
			10.0f,
			line_width);
	}

	{ // map the cuda graphics resource
		ERRCHECK(cudaGraphicsMapResources(1, &cudaPBOResource, 0));
		size_t numBytes = 0;
		ERRCHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_image, &numBytes, cudaPBOResource));
	}

	{ // post processing
		dim3 blockSize(256);
		dim3 gridSize((settings.resolution.x * settings.resolution.y + blockSize.x - 1) / blockSize.x);

		if (settings.accumulationMode == AccumulationMode::TAA) {
			numProgressiveFrames++;
			glm::mat4 modelMat = glm::affineInverse(settings.cameraViewMatrix);
			glm::mat4 deltaMatrix = settings.previousViewProjectionMatrix * modelMat * glm::inverse(settings.cameraProjectionMatrix);

			KERNEL_LAUNCH(temporalRender, gridSize, blockSize,
				d_image,
				d_colorBuffer[a],
				d_colorBuffer[b],
				d_depthBuffer[a],
				d_depthBuffer[b],
				deltaMatrix,
				0.9f,
				settings.resolution.x,
				settings.resolution.y,
				numProgressiveFrames);
		}
		else {
			if (settings.cameraMoved || settings.accumulationMode == AccumulationMode::None)
				numProgressiveFrames = 1;
			else
				numProgressiveFrames++;

			KERNEL_LAUNCH(progressiveRender, gridSize, blockSize,
				d_image,
				d_colorBuffer[a],
				d_colorBuffer[b],
				settings.resolution.x,
				settings.resolution.y,
				numProgressiveFrames);
		}
	}


	ERRCHECK(cudaDeviceSynchronize());

	if (settings.take_pmf_screenshot) {
		std::string timestamp = getCurrentDateTime();
		std::string rel_dir = "out/float_frames";
		std::string rel_path = rel_dir + "/frame_" + timestamp + ".pfm";
		std::string abs_path = resolveProjectPath(rel_path);

		create_directory(resolveProjectPath(rel_dir));

		export_pfm_from_buffer(
			d_colorBuffer[a],
			settings.resolution.x,
			settings.resolution.y,
			numProgressiveFrames,
			abs_path
		);

		settings.take_pmf_screenshot = false;
	}


	if (saveNextFrame) {
		size_t imageSize = settings.resolution.x * settings.resolution.y * 3 * sizeof(unsigned char);
		std::vector<unsigned char> h_image_data(imageSize);

		ERRCHECK(cudaMemcpy(h_image_data.data(), d_image, imageSize, cudaMemcpyDeviceToHost));

		save_image_to_disk(settings, h_image_data.data(), next_filename);
		saveNextFrame = false;
	}


	{ // Unmap the resource so OpenGL can use it.
		ERRCHECK(cudaGraphicsUnmapResources(1, &cudaPBOResource, 0));
	}
}

static void write_pfm(const std::string& path,
	const float* data, int W, int H)
{
	std::ofstream out(path, std::ios::binary);
	out << "PF\n" << W << " " << H << "\n" << "-1.0\n";

	const int stride = W * 3;
	for (int y = 0; y < H; ++y) {
		const float* row = data + y * stride;
		out.write(reinterpret_cast<const char*>(row),
			stride * sizeof(float));
	}
}

void RenderServer::export_pfm_from_buffer(
	const glm::vec3* d_accumBuffer,
	int width,
	int height,
	int numFrames,
	const std::string& outputPath
) {

	const int N = width * height; 
	std::vector<float> h_float(3 * N);
	std::vector<glm::vec3> h_color(N); 
	ERRCHECK(cudaMemcpy(h_color.data(), d_accumBuffer, N * sizeof(glm::vec3), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; ++i) {
		glm::vec3 c = h_color[i] / float(numFrames);
		h_float[3 * i + 0] = c.r; 
		h_float[3 * i + 1] = c.g; 
		h_float[3 * i + 2] = c.b;
	} std::string timestamp = getCurrentDateTime(); 
	std::string rel_dir = "out/float_frames";
	std::string rel_path = rel_dir + "/frame_" + timestamp + ".pfm";
	std::string abs_path = resolveProjectPath(rel_path);
	create_directory(resolveProjectPath(rel_dir)); 
	write_pfm(abs_path, h_float.data(), width, height); 
	std::cout << "Saved float PFM to: " << abs_path << std::endl;
}


GLuint RenderServer::get_pbo() const {
	return pbo;
}

void RenderServer::get_image_data(const RenderSettings& settings, unsigned char*& outBuffer) {
	size_t imageSize = settings.resolution.x * settings.resolution.y * 3 * sizeof(unsigned char);

	ERRCHECK(cudaGraphicsMapResources(1, &cudaPBOResource, 0));

	size_t mappedSize = 0;
	void* pboDevicePtr = nullptr;
	ERRCHECK(cudaGraphicsResourceGetMappedPointer(&pboDevicePtr, &mappedSize, cudaPBOResource));

	if (mappedSize < imageSize) {
		std::cerr << "Mapped size smaller than expected image size." << std::endl;
		ERRCHECK(cudaGraphicsUnmapResources(1, &cudaPBOResource, 0));
		return;
	}

	ERRCHECK(cudaMemcpy(outBuffer, pboDevicePtr, imageSize, cudaMemcpyDeviceToHost));
	ERRCHECK(cudaGraphicsUnmapResources(1, &cudaPBOResource, 0));
}

void RenderServer::set_save_next_frame(bool save, std::string file_name) {
	saveNextFrame = save;
	next_filename = file_name;
}


void RenderServer::save_image_to_disk(const RenderSettings& settings, unsigned char* image_data, std::string custom_name) {
	const int channels = 3;
	const int row_size_bytes = settings.resolution.x * channels;
	std::vector<unsigned char> temp_row(row_size_bytes);
	for (int y = 0; y < settings.resolution.y / 2; ++y) {
		unsigned char* top_row = image_data + y * row_size_bytes;
		unsigned char* bottom_row = image_data + (settings.resolution.y - 1 - y) * row_size_bytes;
		std::memcpy(temp_row.data(), top_row, row_size_bytes);
		std::memcpy(top_row, bottom_row, row_size_bytes);
		std::memcpy(bottom_row, temp_row.data(), row_size_bytes);
	}

	std::string timestamp = getCurrentDateTime();
	std::string relative_dir = "out/screenshots";
	
	std::string relative_path = relative_dir + "/screenshot_" + timestamp + ".png";
	if (!custom_name.empty()) {
		relative_dir += "/eval/" + settings.model_name;
		relative_path = relative_dir + "/" + custom_name + ".png";
	}
		

	std::string absolute_path =  resolveProjectPath(relative_path);	

	if (!create_directory(resolveProjectPath(relative_dir))) {
		std::cerr << "ERROR: Failed to ensure directory structure for screenshots." << std::endl;
		return;
	}

	std::cout << "Saving screenshot to: " << absolute_path << std::endl;

	int success = stbi_write_png(
		absolute_path.c_str(),
		settings.resolution.x,
		settings.resolution.y,
		3,
		image_data,
		settings.resolution.x * 3
	);

	if (success) {
		std::cout << "Screenshot saved successfully!" << std::endl;
	}
	else {
		std::cerr << "ERROR: Failed to save screenshot to " << absolute_path << std::endl;
	}
}