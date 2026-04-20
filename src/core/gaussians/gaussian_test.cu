#include "gaussian_test.h"





void GaussianTest::createXYZGizmoScene(std::vector<CPUGaussian>& gaussians) {

	//// Gizmo
	//gaussians.push_back(CreateGaussian(glm::vec3(1.5f, 0.0f, 0.0f), glm::vec3(1.0f, 0.1f, 0.1f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0xFF0000FF));
	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 1.5f, 0.0f), glm::vec3(0.1f, 1.0f, 0.1f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x00FF00FF));
	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, 1.5f), glm::vec3(0.1f, 0.1f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x0000FFFF));

	//gaussians.push_back(CreateGaussian(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.05f, 0.05f, 0.5f), glm::vec4(1.0f, 0.0f, 0.0f, 0.0f), 0x00FFFFFF));
	//gaussians.push_back(CreateGaussian(glm::vec3(1.0f, 2.0f, 1.0f), glm::vec3(0.05f, 0.05f, 0.5f), glm::vec4(0.707f, 0.0f, 0.0f, 0.707f), 0xFF00FFFF));
	//gaussians.push_back(CreateGaussian(glm::vec3(1.0f, 3.0f, 1.0f), glm::vec3(0.05f, 0.05f, 0.5f), glm::vec4(0.414f, 0.0f, 0.0f, 0.910f), 0x00FFFFFF));
	//gaussians.push_back(CreateGaussian(glm::vec3(1.0f, 4.0f, 1.0f), glm::vec3(0.05f, 0.05f, 0.5f), glm::vec4(0.766f, 0.0f, 0.0f, -0.643f), 0xFF00FFFF));

	//gaussians.push_back(CreateGaussian(glm::vec3(1.0f, -1.0f, 1.0f), glm::vec3(0.05f, 0.05f, 0.5f), glm::vec4(0.707f, 0.0f, 0.707f, 0.0f), 0x00FFFFFF));
	//gaussians.push_back(CreateGaussian(glm::vec3(1.0f, -2.0f, 1.0f), glm::vec3(0.05f, 0.05f, 0.5f), glm::vec4(0.653f, 0.271f, 0.653f, 0.271f), 0xFF00FFFF));
	//gaussians.push_back(CreateGaussian(glm::vec3(1.0f, -3.0f, 1.0f), glm::vec3(0.05f, 0.05f, 0.5f), glm::vec4(0.271f, 0.653f, 0.271f, 0.653f), 0x00FFFFFF));
	//gaussians.push_back(CreateGaussian(glm::vec3(1.0f, -4.0f, 1.0f), glm::vec3(0.05f, 0.05f, 0.5f), glm::vec4(0.0, 0.707f, 0.0f, 0.707f), 0xFF00FFFF));


	//idk
	gaussians.push_back(CreateGaussian(glm::vec3(10.0f, 0.0f, 0.0f), glm::vec3(0.1f, 0.1f, 0.1f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0xFF0000FF));
	gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.25f, 0.25f, 0.25f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x00FF00FF));
	gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x0000FFFF));
	gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0xFFFFFFFF));


	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0001f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x00FF000A));
	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0001f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x00FF000A));
	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, -10.0f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x00800080));
	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, -15.0f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x800000FF));
	// 
	gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, -20.0f), glm::vec3(0.01f, 2.0f, 0.5f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x00FF00FF));
	gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, -30.0f), glm::vec3(0.001f, 0.5f, 0.001f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0xFF00FFFF));
	gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, -40.0f), glm::vec3(2.0f, 0.5f, 0.5f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x00FFFF0A));
	gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, -50.0f), glm::vec3(2.0f, 2.0f, 2.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0xFF00FFFF));
	gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, -60.0f), glm::vec3(0.001f, 4.0f, 4.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x00FFFFFF));

	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, -20.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0xFF0000FF));
	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, -10.0f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0xFFFFFFFF));
	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0x0000FFFF));
	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, 20.0f), glm::vec3(2.0f, 2.0f, 2.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0xFFFFFFFF));
	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, 40.0f), glm::vec3(4.0f, 4.0f, 4.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0xFFFFFFFF));
	//gaussians.push_back(CreateGaussian(glm::vec3(0.0f, 0.0f, 40.0f), glm::vec3(4.0f, 4.0f, 4.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0xFFFFFFFF));
}



// Function to normalize a float3 vector
float3 GaussianTest::normalize(float3 v) {
	float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return { v.x / length, v.y / length, v.z / length };
}

// Function to compute the cross product of two float3 vectors
float3 GaussianTest::cross(const float3& a, const float3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

// Function to compute the dot product of two float3 vectors
float GaussianTest::dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

void GaussianTest::createSphereScene(std::vector<CPUGaussian>& gaussians, float r) {
	//const int numLatitude = 3;
	//const int numLongitude = 6;
	//float3 scale = glm::vec3(0.05f, 0.5f, 0.05f);

	//for (int i = 0; i <= numLatitude; ++i) {
	//	float theta = i * 3.141593f / numLatitude;
	//	for (int j = 0; j < numLongitude; ++j) {
	//		float phi = j * 2 * 3.141593f / numLongitude;
	//		float x = r * sin(theta) * cos(phi);
	//		float y = r * sin(theta) * sin(phi);
	//		float z = r * cos(theta);
	//		float3 position = glm::vec3(x, y, z);

	//		// Calculate the quaternion to rotate the longest axis towards the origin
	//		float3 direction = normalize(glm::vec3(-x, -y, -z));
	//		float4 rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // Placeholder for actual quaternion calculation

	//		// Ensure the rotation quaternion points the longest axis towards the origin
	//		float3 forward = glm::vec3(0.0f, 1.0f, 0.0f);
	//		float dotProd = dot(forward, direction);
	//		float3 axis = cross(forward, direction);
	//		float angle = acos(dotProd);
	//		float s = sin(angle / 2);
	//		rotation = glm::vec4(axis.x * s, axis.y * s, axis.z * s, cos(angle / 2));

	//		//gaussians.push_back(CreateGaussian(position, scale, rotation, 0xFFFFFFFF)); // Using a fixed color for simplicity
	//	}
	//}
}


void GaussianTest::generate() {
	std::vector<CPUGaussian> gaussians;
	createXYZGizmoScene(gaussians);
	//createSphereScene(gaussians, 4.0f);
	//GaussianLoader::writePlyFile("../GaussianSplattingScenes/custom/test_scene.ply", gaussians);
	GaussianLoader::writePlyFile("../GaussianSplattingScenes/original_gaussian_splatting/test/point_cloud/iteration_30000/point_cloud.ply", gaussians);
}


