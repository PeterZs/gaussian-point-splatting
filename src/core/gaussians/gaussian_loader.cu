#include "gaussian_loader.cuh"
#include <algorithm>
#include <numeric>  
#include <limits>   
#include <execution>
#include "utils/io.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

#include "config.h"

constexpr bool FORCE_IN_PLACE = true;

int D = 3;
int SH_N = (D + 1) * (D + 1);

/// ---------------------------------------- Support for multiple header types ----------------------------------------

#pragma pack(push, 1)
struct PLYFileGaussianFull {
	float mean[3];      // x, y, z
	float normal[3];    // nx, ny, nz
	float sh[48];       // f_dc_0..2 + f_rest_0..44
	float opacity;
	float scale[3];
	float rotation[4];
};
#pragma pack(pop)

#pragma pack(push, 1)
struct PLYFileGaussianFullNoNormal {
	float mean[3];      // x, y, z
	float sh[48];       // f_dc_0..2 + f_rest_0..44
	float opacity;
	float scale[3];
	float rotation[4];
};
#pragma pack(pop)

#pragma pack(push, 1)
struct PLYFileGaussianMinimal {
	float mean[3];      // x, y, z
	float f_dc[3];      // f_dc_0..2
	float opacity;
	float scale[3];
	float rotation[4];
};
#pragma pack(pop)

static_assert(sizeof(PLYFileGaussianFull) == 248, "Unexpected size for FULL");
static_assert(sizeof(PLYFileGaussianFullNoNormal) == 236, "Unexpected size for FULL NO NORMAL");
static_assert(sizeof(PLYFileGaussianMinimal) == 56, "Unexpected size for MINIMAL");

inline CPUGaussian toCPUGaussian(const PLYFileGaussianFull& fg) {
	CPUGaussian g;
	g.mean = glm::vec3(fg.mean[0], fg.mean[1], fg.mean[2]);
	g.opacity = glm::clamp(1.0f / (1.0f + std::exp(-fg.opacity)), 0.0f, 1.0f);
	g.scale = glm::exp(glm::vec3(fg.scale[0], fg.scale[1], fg.scale[2]));
	g.rotation = glm::normalize(glm::vec4(fg.rotation[0], fg.rotation[1],
		fg.rotation[2], fg.rotation[3]));
	g.sh[0] = fg.sh[0]; g.sh[1] = fg.sh[1]; g.sh[2] = fg.sh[2];
#ifndef DISABLE_SPHERICAL_HARMONICS	
	for (int j = 1; j < SH_N; ++j) {
		g.sh[j * 3 + 0] = fg.sh[(j - 1) + 3];
		g.sh[j * 3 + 1] = fg.sh[(j - 1) + SH_N + 2];
		g.sh[j * 3 + 2] = fg.sh[(j - 1) + 2 * SH_N + 1];
	}
#endif
	return g;
}

inline CPUGaussian toCPUGaussian(const PLYFileGaussianFullNoNormal& fg) {
	CPUGaussian g;
	g.mean = glm::vec3(fg.mean[0], fg.mean[1], fg.mean[2]);
	g.opacity = glm::clamp(1.0f / (1.0f + std::exp(-fg.opacity)), 0.0f, 1.0f);
	g.scale = glm::exp(glm::vec3(fg.scale[0], fg.scale[1], fg.scale[2]));
	g.rotation = glm::normalize(glm::vec4(fg.rotation[0], fg.rotation[1],
		fg.rotation[2], fg.rotation[3]));
	g.sh[0] = fg.sh[0]; g.sh[1] = fg.sh[1]; g.sh[2] = fg.sh[2];
#ifndef DISABLE_SPHERICAL_HARMONICS
	for (int j = 1; j < SH_N; ++j) {
		g.sh[j * 3 + 0] = fg.sh[(j - 1) + 3];
		g.sh[j * 3 + 1] = fg.sh[(j - 1) + SH_N + 2];
		g.sh[j * 3 + 2] = fg.sh[(j - 1) + 2 * SH_N + 1];
	}
#endif
	return g;
}

inline CPUGaussian toCPUGaussian(const PLYFileGaussianMinimal& fg) {
	CPUGaussian g;
	g.mean = glm::vec3(fg.mean[0], fg.mean[1], fg.mean[2]);
	g.opacity = glm::clamp(1.0f / (1.0f + std::exp(-fg.opacity)), 0.0f, 1.0f);
	g.scale = glm::exp(glm::vec3(fg.scale[0], fg.scale[1], fg.scale[2]));
	g.rotation = glm::normalize(glm::vec4(fg.rotation[0], fg.rotation[1],
		fg.rotation[2], fg.rotation[3]));
	g.sh[0] = fg.f_dc[0]; g.sh[1] = fg.f_dc[1]; g.sh[2] = fg.f_dc[2];
#ifndef DISABLE_SPHERICAL_HARMONICS // if the renderer expects SH coefficients, just feed it 0.
	for (int j = 1; j < SH_N; ++j) {
		g.sh[j * 3 + 0] = 0.0f;
		g.sh[j * 3 + 1] = 0.0f;
		g.sh[j * 3 + 2] = 0.0f;
	}
#endif
	return g;
}

struct SchemaSpec {
	size_t recordSize;
	CPUGaussian(*convert)(const void*);
};
//function that returns a function to convert gaussians of different headers into the one we use.
inline SchemaSpec makeSchemaSpec(const std::vector<std::string>& props,
	const std::vector<std::string>& headerLines,
	bool& ok)
{
	auto has = [&](const char* n) {
		return std::find(props.begin(), props.end(), n) != props.end();
	};
	bool hasNormals = has("nx") && has("ny") && has("nz");
	bool hasRest = std::any_of(props.begin(), props.end(),
		[](const std::string& p) { return p.rfind("f_rest_", 0) == 0; });
	bool hasDc = has("f_dc_0") && has("f_dc_1") && has("f_dc_2");
	bool full = hasNormals && hasRest && hasDc;
	bool fullNoNormal = !hasNormals && hasRest && hasDc;
	bool minimal = hasDc && !hasNormals && !hasRest;


	if (fullNoNormal) {
		ok = true;
		return SchemaSpec{
			sizeof(PLYFileGaussianFullNoNormal),
			[](const void* rec) -> CPUGaussian {
				return toCPUGaussian(*reinterpret_cast<const PLYFileGaussianFullNoNormal*>(rec));
			}
		};
	}
	else if (full) {
		ok = true;
		return SchemaSpec{
			sizeof(PLYFileGaussianFull),
			[](const void* rec) -> CPUGaussian {
				return toCPUGaussian(*reinterpret_cast<const PLYFileGaussianFull*>(rec));
			}
		};
	}
	else if (minimal) {
		ok = true;
		return SchemaSpec{
			sizeof(PLYFileGaussianMinimal),
			[](const void* rec) -> CPUGaussian {
				return toCPUGaussian(*reinterpret_cast<const PLYFileGaussianMinimal*>(rec));
			}
		};
	}

	ok = false;
	return SchemaSpec{ 0, nullptr };
}



/// ---------------------------------------- Loading ----------------------------------------

// Safe clamp for division when max == min on an axis
inline glm::vec3 safe_div(const glm::vec3& num, const glm::vec3& den) {
	return glm::vec3(
		den.x != 0.0f ? num.x / den.x : 0.0f,
		den.y != 0.0f ? num.y / den.y : 0.0f,
		den.z != 0.0f ? num.z / den.z : 0.0f
	);
}

// Compute 3D Morton code (21 bits per axis) from [0,1] normalized position
inline uint64_t morton3D21(const glm::ivec3& xyz) {
	uint64_t code = 0;
	for (int bit = 0; bit < 21; ++bit) {
		code |= (uint64_t(xyz.x & (1 << bit)) << (2 * bit + 0));
		code |= (uint64_t(xyz.y & (1 << bit)) << (2 * bit + 1));
		code |= (uint64_t(xyz.z & (1 << bit)) << (2 * bit + 2));
	}
	return code;
}

// Cross-platform free memory query
static size_t getFreeSystemMemoryBytes() {
#ifdef _WIN32
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	if (GlobalMemoryStatusEx(&status)) {
		return static_cast<size_t>(status.ullAvailPhys);
	}
	return 0;
#else
	struct sysinfo info;
	if (sysinfo(&info) == 0) {
		return static_cast<size_t>(info.freeram) * info.mem_unit;
	}
	return 0;
#endif
}


bool GaussianLoader::loadPlyFile(const std::string& filename, std::vector<CPUGaussian>& gaussians, std::vector<PlyComment>& comments, const bool sort_morton_order) {
	std::string path = resolveProjectPath(filename);
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) { std::cerr << "Failed to open file: " << path << "\n"; return false; }

	std::cout << "Loading file: " << path << std::endl;

	comments.clear();
	size_t count = 0;
	std::vector<std::string> headerLines;
	std::vector<std::string> props;

	// Parse header (as you already do)
	{
		std::string line;
		while (std::getline(file, line)) {
			headerLines.push_back(line);
			auto first = line.find_first_not_of(" \t\r\n");
			if (first != std::string::npos) line = line.substr(first);

			if (line.rfind("comment", 0) == 0) {
				comments.push_back({ line.size() > 8 ? line.substr(8) : std::string() });
			}
			if (line.rfind("element vertex", 0) == 0) {
				std::istringstream iss(line);
				std::string s1, s2;
				iss >> s1 >> s2 >> count;
			}
			if (line.rfind("property", 0) == 0) {
				std::istringstream iss(line);
				std::string s_prop, type, name;
				iss >> s_prop >> type >> name;
				if (!name.empty()) props.push_back(name);
			}
			if (line == "end_header") break;
		}
	}

	if (count == 0) { std::cerr << "Header missing or zero vertex count.\n"; return false; }

	bool ok = false;
	SchemaSpec spec = makeSchemaSpec(props, headerLines, ok);
	if (!ok) {
		std::cerr << "Unsupported PLY schema. Full header follows:\n";
		for (auto& l : headerLines) std::cerr << l << "\n";
		gaussians.clear();
		return false;
	}

	gaussians.resize(count);

	size_t required = count * spec.recordSize;
	size_t freeMem = getFreeSystemMemoryBytes();

	if (freeMem > 2 * required) {
		// Bulk read into raw byte buffer, then convert in parallel
		std::vector<uint8_t> buffer(required);
		file.read(reinterpret_cast<char*>(buffer.data()), required);
		if (!file) { std::cerr << "Short read in bulk mode.\n"; return false; }

		// Parallel conversion
		std::for_each(std::execution::par_unseq,
			gaussians.begin(), gaussians.end(),
			[&](CPUGaussian& g) {
			size_t i = static_cast<size_t>(&g - gaussians.data());
			const void* rec = buffer.data() + i * spec.recordSize;
			g = spec.convert(rec);
		});
	}
	else {
		// Streaming read: record-by-record
		std::vector<uint8_t> rec(spec.recordSize);
		for (size_t i = 0; i < count; ++i) {
			file.read(reinterpret_cast<char*>(rec.data()), spec.recordSize);
			if (!file) {
				std::cerr << "Short read: expected " << count << " records, got " << i << "\n";
				gaussians.resize(i);
				break;
			}
			gaussians[i] = spec.convert(rec.data());
		}
	}

	if (sort_morton_order) {
		// Parallel bounding box (single pass)
		struct MinMax { glm::vec3 min; glm::vec3 max; };

		auto combine = [](const MinMax& a, const MinMax& b) -> MinMax {
			return MinMax{ glm::min(a.min, b.min), glm::max(a.max, b.max) };
		};
		auto transform = [](const CPUGaussian& g) -> MinMax {
			return MinMax{ g.mean, g.mean };
		};

		MinMax mm = std::transform_reduce(
			std::execution::par_unseq,
			gaussians.begin(), gaussians.end(),
			MinMax{ glm::vec3(FLT_MAX), glm::vec3(-FLT_MAX) },
			combine,
			transform
		);

		glm::vec3 minn = mm.min;
		glm::vec3 maxx = mm.max;

		const size_t N = gaussians.size();

		std::vector<uint64_t> morton(N);

		std::for_each(std::execution::par_unseq,
			morton.begin(), morton.end(),
			[&](size_t i) {
			glm::vec3 rel = safe_div(gaussians[i].mean - minn, maxx - minn);
			glm::vec3 scaled = rel * float((1 << 21) - 1);
			glm::ivec3 xyz = glm::ivec3(scaled);
			morton[i] = morton3D21(xyz);
		});

		// Sort indices by morton in parallel
		std::vector<size_t> idx(N);
		std::iota(idx.begin(), idx.end(), 0);

		std::sort(std::execution::par_unseq,
			idx.begin(), idx.end(),
			[&](size_t a, size_t b) {
			return morton[a] < morton[b];
		});

		// Apply permutation to gaussians in place (cycle decomposition)
		std::vector<bool> visited(N, false);
		for (size_t i = 0; i < N; ++i) {
			if (visited[i] || idx[i] == i) continue;
			size_t j = i;
			CPUGaussian tmp = std::move(gaussians[i]);
			while (!visited[j]) {
				visited[j] = true;
				size_t k = idx[j];
				if (k == i) {
					gaussians[j] = std::move(tmp);
					break;
				}
				else {
					gaussians[j] = std::move(gaussians[k]);
					j = k;
				}
			}
		}
	}

	//for (auto& g : gaussians) {
	//	g.opacity = 0.5f;
	//}

	file.close();
	return true;
}


void GaussianLoader::printGaussian(const CPUGaussian& gaussian) {
	std::cout << "Gaussian:\n";
	std::cout << "Mean: (" << gaussian.mean.x << ", " << gaussian.mean.y << ", " << gaussian.mean.z << ")\n";
	std::cout << "f_dc: (" << gaussian.sh[0] << ", " << gaussian.sh[1] << ", " << gaussian.sh[2] << ")\n";
	std::cout << "f_rest: ";
	for (int i = 0; i < 45; ++i) {
		std::cout << gaussian.sh[i + 1];
		if (i < 44) std::cout << ", ";
	}
	std::cout << "\n";
	std::cout << "Opacity: " << gaussian.opacity << "\n";
	std::cout << "Scale: (" << gaussian.scale.x << ", " << gaussian.scale.y << ", " << gaussian.scale.z << ")\n";
	std::cout << "Rotation: (" << gaussian.rotation.x << ", " << gaussian.rotation.y << ", " << gaussian.rotation.z << ", " << gaussian.rotation.w << ")\n";
}

std::string makeHeaderMinimal(size_t count, const std::vector<PlyComment>& comments) {
	std::ostringstream header;
	header << "ply\n"
		<< "format binary_little_endian 1.0\n";
	for (const auto& c : comments) {
		header << "comment " << c.text << "\n";
	}
	header << "element vertex " << count << "\n"
		<< "property float x\n"
		<< "property float y\n"
		<< "property float z\n"
		<< "property float f_dc_0\n"
		<< "property float f_dc_1\n"
		<< "property float f_dc_2\n"
		<< "property float opacity\n"
		<< "property float scale_0\n"
		<< "property float scale_1\n"
		<< "property float scale_2\n"
		<< "property float rot_0\n"
		<< "property float rot_1\n"
		<< "property float rot_2\n"
		<< "property float rot_3\n"
		<< "end_header\n";
	return header.str();
}

std::string makeHeaderFullNoNormal(size_t count, const std::vector<PlyComment>& comments) {
	std::ostringstream header;
	header << "ply\n"
		<< "format binary_little_endian 1.0\n";
	for (const auto& c : comments) {
		header << "comment " << c.text << "\n";
	}
	header << "element vertex " << count << "\n"
		<< "property float x\n"
		<< "property float y\n"
		<< "property float z\n"
		<< "property float f_dc_0\n"
		<< "property float f_dc_1\n"
		<< "property float f_dc_2\n";
	for (int i = 0; i < 45; ++i) {
		header << "property float f_rest_" << i << "\n";
	}
	header << "property float opacity\n"
		<< "property float scale_0\n"
		<< "property float scale_1\n"
		<< "property float scale_2\n"
		<< "property float rot_0\n"
		<< "property float rot_1\n"
		<< "property float rot_2\n"
		<< "property float rot_3\n"
		<< "end_header\n";
	return header.str();
}

std::string makeHeaderFull(size_t count, const std::vector<PlyComment>& comments) {
	std::ostringstream header;
	header << "ply\n"
		<< "format binary_little_endian 1.0\n";
	for (const auto& c : comments) {
		header << "comment " << c.text << "\n";
	}
	header << "element vertex " << count << "\n"
		<< "property float x\n"
		<< "property float y\n"
		<< "property float z\n"
		<< "property float nx\n"
		<< "property float ny\n"
		<< "property float nz\n"
		<< "property float f_dc_0\n"
		<< "property float f_dc_1\n"
		<< "property float f_dc_2\n";
	for (int i = 0; i < 45; ++i) {
		header << "property float f_rest_" << i << "\n";
	}
	header << "property float opacity\n"
		<< "property float scale_0\n"
		<< "property float scale_1\n"
		<< "property float scale_2\n"
		<< "property float rot_0\n"
		<< "property float rot_1\n"
		<< "property float rot_2\n"
		<< "property float rot_3\n"
		<< "end_header\n";
	return header.str();
}


void GaussianLoader::writePlyFile(const std::string& filename,
	const std::vector<CPUGaussian>& gaussians,
	const std::vector<PlyComment>& comments) {
	std::string path = resolveProjectPath(filename);
	std::ofstream file(path, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << path << std::endl;
		return;
	}

#ifdef DISABLE_SPHERICAL_HARMONICS
	std::string header = makeHeaderMinimal(gaussians.size(), comments);
#else
	std::string header = makeHeaderFull(gaussians.size(), comments);
#endif
	file.write(header.c_str(), header.size());

	for (const auto& g : gaussians) {
#ifdef DISABLE_SPHERICAL_HARMONICS
		// Pack Minimal
		PLYFileGaussianMinimal out{};
		uint32_t sh_color = static_cast<uint32_t>(g.sh[0]) << 24 | static_cast<uint32_t>(g.sh[1]) << 16 | static_cast<uint32_t>(g.sh[2]) << 8 | 0xFFu;
		CudaMath::base_from_color(CudaMath::uintRGBAToVec4(sh_color), out.f_dc);


		out.mean[0] = g.mean.x; out.mean[1] = g.mean.y; out.mean[2] = g.mean.z;
		out.opacity = std::log(g.opacity / (1.0f - g.opacity));
		glm::vec3 inv_scale = glm::log(g.scale);
		out.scale[0] = inv_scale.x; out.scale[1] = inv_scale.y; out.scale[2] = inv_scale.z;
		out.rotation[0] = g.rotation.x; out.rotation[1] = g.rotation.y;
		out.rotation[2] = g.rotation.z; out.rotation[3] = g.rotation.w;
		file.write(reinterpret_cast<const char*>(&out), sizeof(out));
#else
		// Pack FullNoNormal
		PLYFileGaussianFull out{};
		//PLYFileGaussianFullNoNormal out{};
		out.mean[0] = g.mean.x; out.mean[1] = g.mean.y; out.mean[2] = g.mean.z;
		out.sh[0] = g.sh[0]; out.sh[1] = g.sh[1]; out.sh[2] = g.sh[2];
		for (int j = 1; j < SH_N; ++j) {
			out.sh[j * 3 + 0] = g.sh[(j - 1) + 3];
			out.sh[j * 3 + 1] = g.sh[(j - 1) + SH_N + 2];
			out.sh[j * 3 + 2] = g.sh[(j - 1) + 2 * SH_N + 1];
		}
		out.opacity = std::log(g.opacity / (1.0f - g.opacity));
		glm::vec3 inv_scale = glm::log(g.scale);
		out.scale[0] = inv_scale.x; out.scale[1] = inv_scale.y; out.scale[2] = inv_scale.z;
		out.rotation[0] = g.rotation.x; out.rotation[1] = g.rotation.y;
		out.rotation[2] = g.rotation.z; out.rotation[3] = g.rotation.w;
		file.write(reinterpret_cast<const char*>(&out), sizeof(out));
#endif
	}

	file.close();
	std::cout << "Successfully wrote to file: " << path << std::endl;
}


