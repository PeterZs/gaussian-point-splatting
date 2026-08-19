#include "gaussian_loader.cuh"
#include <algorithm>
#include <numeric>
#include <limits>
#include "utils/io.h"

#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

#include "config.h"

int D = 3;
int SH_N = (D + 1) * (D + 1);

/// ---------------------------------------- Packed structs used by the write path ----------------------------------------

#pragma pack(push, 1)
struct PLYFileGaussianFull {
    float mean[3];
    float normal[3];
    float sh[48];
    float opacity;
    float scale[3];
    float rotation[4];
};
#pragma pack(pop)

#pragma pack(push, 1)
struct PLYFileGaussianFullNoNormal {
    float mean[3];
    float sh[48];
    float opacity;
    float scale[3];
    float rotation[4];
};
#pragma pack(pop)

#pragma pack(push, 1)
struct PLYFileGaussianMinimal {
    float mean[3];
    float f_dc[3];
    float opacity;
    float scale[3];
    float rotation[4];
};
#pragma pack(pop)

static_assert(sizeof(PLYFileGaussianFull) == 248, "Unexpected size for FULL");
static_assert(sizeof(PLYFileGaussianFullNoNormal) == 236, "Unexpected size for FULL NO NORMAL");
static_assert(sizeof(PLYFileGaussianMinimal) == 56, "Unexpected size for MINIMAL");

/// ---------------------------------------- Flexible offset-based PLY field descriptor ----------------------------------------

static constexpr size_t OFFSET_MISSING = SIZE_MAX;

struct PlyFieldOffsets {
    // Position
    size_t x = OFFSET_MISSING, y = OFFSET_MISSING, z = OFFSET_MISSING;

    // Spherical harmonics: DC (3 floats) + rest (up to 45 floats)
    size_t f_dc[3] = { OFFSET_MISSING, OFFSET_MISSING, OFFSET_MISSING };
    std::vector<size_t> f_rest; // interleaved in the PLY: f_rest_0..14 (R), 15..29 (G), 30..44 (B)

    // Opacity
    size_t opacity = OFFSET_MISSING;

    // Scale
    size_t scale[3] = { OFFSET_MISSING, OFFSET_MISSING, OFFSET_MISSING };

    // Rotation (quaternion stored as rot_0..3 = w, x, y, z in the original 3DGS convention)
    size_t rot[4] = { OFFSET_MISSING, OFFSET_MISSING, OFFSET_MISSING, OFFSET_MISSING };

    // Total bytes per record
    size_t record_size = 0;

    bool has_rest() const { return !f_rest.empty(); }

    bool is_valid(std::string& err) const {
        if (x == OFFSET_MISSING || y == OFFSET_MISSING || z == OFFSET_MISSING) {
            err = "missing position fields (x/y/z)"; return false;
        }
        if (f_dc[0] == OFFSET_MISSING || f_dc[1] == OFFSET_MISSING || f_dc[2] == OFFSET_MISSING) {
            err = "missing DC spherical harmonic fields (f_dc_0/1/2)"; return false;
        }
        if (opacity == OFFSET_MISSING) { err = "missing opacity field"; return false; }
        for (int i = 0; i < 3; ++i)
            if (scale[i] == OFFSET_MISSING) { err = "missing scale field(s)"; return false; }
        for (int i = 0; i < 4; ++i)
            if (rot[i] == OFFSET_MISSING) { err = "missing rotation field(s)"; return false; }
        return true;
    }
};

// Returns byte width of a PLY scalar type name
static size_t plyTypeSize(const std::string& type) {
    if (type == "float" || type == "float32" || type == "int" || type == "uint" || type == "int32" || type == "uint32") return 4;
    if (type == "double" || type == "float64" || type == "int64" || type == "uint64") return 8;
    if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") return 2;
    if (type == "char" || type == "uchar" || type == "int8" || type == "uint8")  return 1;
    return 4; // safe fallback � all 3DGS fields are float
}

/// Parse the PLY header and fill a PlyFieldOffsets. Returns false on fatal error.
static bool parsePlyHeader(std::istream& file,
    size_t& out_count,
    PlyFieldOffsets& out_offsets,
    std::vector<PlyComment>& out_comments,
    std::vector<std::string>& out_header_lines)
{
    out_count = 0;
    out_comments.clear();
    out_header_lines.clear();
    out_offsets = PlyFieldOffsets{};

    std::string line;
    size_t byte_offset = 0;

    while (std::getline(file, line)) {
        out_header_lines.push_back(line);

        // Trim leading whitespace / CR
        auto first = line.find_first_not_of(" \t\r\n");
        if (first != std::string::npos) line = line.substr(first);
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line.rfind("comment", 0) == 0) {
            out_comments.push_back({ line.size() > 8 ? line.substr(8) : std::string() });
        }
        else if (line.rfind("element vertex", 0) == 0) {
            std::istringstream iss(line);
            std::string s1, s2;
            iss >> s1 >> s2 >> out_count;
        }
        else if (line.rfind("property", 0) == 0) {
            std::istringstream iss(line);
            std::string kw, type, name;
            iss >> kw >> type >> name;
            if (name.empty()) continue;

            size_t field_size = plyTypeSize(type);

            // Map name -> offset slot
            if (name == "x")        out_offsets.x = byte_offset;
            else if (name == "y")        out_offsets.y = byte_offset;
            else if (name == "z")        out_offsets.z = byte_offset;
            else if (name == "f_dc_0")   out_offsets.f_dc[0] = byte_offset;
            else if (name == "f_dc_1")   out_offsets.f_dc[1] = byte_offset;
            else if (name == "f_dc_2")   out_offsets.f_dc[2] = byte_offset;
            else if (name == "opacity")  out_offsets.opacity = byte_offset;
            else if (name == "scale_0")  out_offsets.scale[0] = byte_offset;
            else if (name == "scale_1")  out_offsets.scale[1] = byte_offset;
            else if (name == "scale_2")  out_offsets.scale[2] = byte_offset;
            else if (name == "rot_0")    out_offsets.rot[0] = byte_offset;
            else if (name == "rot_1")    out_offsets.rot[1] = byte_offset;
            else if (name == "rot_2")    out_offsets.rot[2] = byte_offset;
            else if (name == "rot_3")    out_offsets.rot[3] = byte_offset;
            else if (name.rfind("f_rest_", 0) == 0) {
                // Index encoded in name: f_rest_N
                int idx = std::stoi(name.substr(7));
                if (idx >= (int)out_offsets.f_rest.size())
                    out_offsets.f_rest.resize(idx + 1, OFFSET_MISSING);
                out_offsets.f_rest[idx] = byte_offset;
            }
            // nx/ny/nz and any unknown fields: just advance the offset, ignore data

            byte_offset += field_size;
        }
        else if (line == "end_header") {
            break;
        }
    }

    out_offsets.record_size = byte_offset;
    return true;
}

/// Read one float from a raw record byte buffer at the given offset.
/// Returns 0.0f if the offset is OFFSET_MISSING (field absent in file).
static inline float getf(const uint8_t* rec, size_t offset) {
    if (offset == OFFSET_MISSING) return 0.0f;
    float v;
    std::memcpy(&v, rec + offset, sizeof(float));
    return v;
}

/// Convert a raw record into a CPUGaussian using the parsed offsets.
static CPUGaussian recordToCPUGaussian(const uint8_t* rec, const PlyFieldOffsets& off) {
    CPUGaussian g;

    // Position
    g.mean = glm::vec3(getf(rec, off.x), getf(rec, off.y), getf(rec, off.z));

    // Opacity: inverse sigmoid
    g.opacity = glm::clamp(1.0f / (1.0f + std::exp(-getf(rec, off.opacity))), 0.0f, 1.0f);

    // Scale: exp of log-scale
    g.scale = glm::exp(glm::vec3(
        getf(rec, off.scale[0]),
        getf(rec, off.scale[1]),
        getf(rec, off.scale[2])
    ));

    // Rotation: stored as (w, x, y, z) -> glm::vec4(x,y,z,w) convention used elsewhere
    g.rotation = glm::normalize(glm::vec4(
        getf(rec, off.rot[0]),
        getf(rec, off.rot[1]),
        getf(rec, off.rot[2]),
        getf(rec, off.rot[3])
    ));

    // SH � DC band always present
    g.sh[0] = getf(rec, off.f_dc[0]);
    g.sh[1] = getf(rec, off.f_dc[1]);
    g.sh[2] = getf(rec, off.f_dc[2]);

#ifndef DISABLE_SPHERICAL_HARMONICS
    // f_rest in PLY is stored channel-interleaved:
    //   f_rest_0..14  -> R coefficients (bands 1-3)
    //   f_rest_15..29 -> G coefficients
    //   f_rest_30..44 -> B coefficients
    const int rest_per_channel = SH_N - 1; // 15 for degree-3
    const int total_expected = 3 * rest_per_channel;
    const bool have_rest = (int)off.f_rest.size() >= total_expected;

    for (int j = 1; j < SH_N; ++j) {
        int rest_idx = j - 1;
        g.sh[j * 3 + 0] = have_rest ? getf(rec, off.f_rest[rest_idx]) : 0.0f;
        g.sh[j * 3 + 1] = have_rest ? getf(rec, off.f_rest[rest_idx + rest_per_channel]) : 0.0f;
        g.sh[j * 3 + 2] = have_rest ? getf(rec, off.f_rest[rest_idx + 2 * rest_per_channel]) : 0.0f;
    }
#endif

    return g;
}

/// ---------------------------------------- Shared helpers ----------------------------------------

inline glm::vec3 safe_div(const glm::vec3& num, const glm::vec3& den) {
    return glm::vec3(
        den.x != 0.0f ? num.x / den.x : 0.0f,
        den.y != 0.0f ? num.y / den.y : 0.0f,
        den.z != 0.0f ? num.z / den.z : 0.0f
    );
}

inline uint64_t morton3D21(const glm::ivec3& xyz) {
    uint64_t code = 0;
    for (int bit = 0; bit < 21; ++bit) {
        code |= (uint64_t(xyz.x & (1 << bit)) << (2 * bit + 0));
        code |= (uint64_t(xyz.y & (1 << bit)) << (2 * bit + 1));
        code |= (uint64_t(xyz.z & (1 << bit)) << (2 * bit + 2));
    }
    return code;
}

static size_t getFreeSystemMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) return static_cast<size_t>(status.ullAvailPhys);
    return 0;
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) return static_cast<size_t>(info.freeram) * info.mem_unit;
    return 0;
#endif
}

/// ---------------------------------------- Loading ----------------------------------------

bool GaussianLoader::loadPlyFile(const std::string& filename,
    std::vector<CPUGaussian>& gaussians,
    std::vector<PlyComment>& comments,
    const bool sort_morton_order)
{
    std::string path = resolveProjectPath(filename);
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << "\n";
        return false;
    }

    std::cout << "Loading file: " << path << "\n";

    size_t count = 0;
    PlyFieldOffsets offsets;
    std::vector<std::string> headerLines;

    if (!parsePlyHeader(file, count, offsets, comments, headerLines)) {
        std::cerr << "Failed to parse PLY header.\n";
        return false;
    }

    if (count == 0) {
        std::cerr << "Header missing or zero vertex count.\n";
        return false;
    }

    // Validate that all required fields were found
    {
        std::string err;
        if (!offsets.is_valid(err)) {
            std::cerr << "Unsupported PLY schema: " << err << "\nFull header:\n";
            for (auto& l : headerLines) std::cerr << l << "\n";
            gaussians.clear();
            return false;
        }
    }

#ifndef DISABLE_SPHERICAL_HARMONICS
    if (!offsets.has_rest()) {
        std::cerr << "Warning: PLY has no f_rest fields -> higher-order SH coefficients will be zero.\n";
    }
#endif

    gaussians.resize(count);

    const size_t required = count * offsets.record_size;
    const size_t freeMem = getFreeSystemMemoryBytes();

    if (freeMem > 2 * required) {
        // Bulk read -> parallel conversion
        std::vector<uint8_t> buffer(required);
        file.read(reinterpret_cast<char*>(buffer.data()), required);
        if (!file) {
            std::cerr << "Short read in bulk mode.\n";
            return false;
        }

        thrust::for_each(thrust::host,
            gaussians.begin(), gaussians.end(),
            [&](CPUGaussian& g) {
            size_t i = static_cast<size_t>(&g - gaussians.data());
            const uint8_t* rec = buffer.data() + i * offsets.record_size;
            g = recordToCPUGaussian(rec, offsets);
        });
    }
    else {
        // Streaming: record by record
        std::vector<uint8_t> rec(offsets.record_size);
        for (size_t i = 0; i < count; ++i) {
            file.read(reinterpret_cast<char*>(rec.data()), offsets.record_size);
            if (!file) {
                std::cerr << "Short read: expected " << count << " records, got " << i << "\n";
                gaussians.resize(i);
                break;
            }
            gaussians[i] = recordToCPUGaussian(rec.data(), offsets);
        }
    }

    if (sort_morton_order && !PlyCommentUtils::hasBVHSorted(comments)) {
        std::cout << "Sorting morton order" << std::endl;
        struct MinMax { glm::vec3 min; glm::vec3 max; };
        auto combine = [](const MinMax& a, const MinMax& b) -> MinMax {
            return MinMax{ glm::min(a.min, b.min), glm::max(a.max, b.max) };
        };
        auto transform = [](const CPUGaussian& g) -> MinMax {
            return MinMax{ g.mean, g.mean };
        };
        MinMax mm = thrust::transform_reduce(thrust::host,
            gaussians.begin(), gaussians.end(),
            transform,
            MinMax{ glm::vec3(FLT_MAX), glm::vec3(-FLT_MAX) },
            combine
        );

        const size_t N = gaussians.size();

        // Compute morton codes on host
        std::vector<uint64_t> morton(N);
        thrust::for_each(thrust::host,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(N),
            [&](size_t i) {
            glm::vec3 rel = safe_div(gaussians[i].mean - mm.min, mm.max - mm.min);
            glm::vec3 scaled = rel * float((1 << 21) - 1);
            morton[i] = morton3D21(glm::ivec3(scaled));
        });

        // Sorted indices, back on host before we leave this scope
        std::vector<uint32_t> idx(N);

        // GPU sort in its own scope — all device memory freed on exit
        {
            thrust::device_vector<uint64_t> d_morton(morton); // upload
            thrust::device_vector<uint32_t> d_idx(N);
            thrust::sequence(d_idx.begin(), d_idx.end());     // 0,1,2,...,N-1

            thrust::sort_by_key(thrust::device,
                d_morton.begin(), d_morton.end(),
                d_idx.begin());

            thrust::copy(d_idx.begin(), d_idx.end(), idx.begin()); // download
        } // d_morton and d_idx destroyed here, GPU memory released

        morton.clear();
        morton.shrink_to_fit(); // free host morton buffer too before permutation

        // In-place permutation via cycle decomposition
        std::vector<bool> visited(N, false);
        for (size_t i = 0; i < N; ++i) {
            if (visited[i] || idx[i] == i) continue;
            size_t j = i;
            CPUGaussian tmp = std::move(gaussians[i]);
            while (!visited[j]) {
                visited[j] = true;
                size_t k = idx[j];
                if (k == i) { gaussians[j] = std::move(tmp); break; }
                else { gaussians[j] = std::move(gaussians[k]); j = k; }
            }
        }
    }
    file.close();
    return true;
}

/// ---------------------------------------- Debug print ----------------------------------------

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
    std::cout << "Rotation: (" << gaussian.rotation.x << ", " << gaussian.rotation.y << ", "
        << gaussian.rotation.z << ", " << gaussian.rotation.w << ")\n";
}

/// ---------------------------------------- Header writers ----------------------------------------

std::string makeHeaderMinimal(size_t count, const std::vector<PlyComment>& comments) {
    std::ostringstream header;
    header << "ply\n" << "format binary_little_endian 1.0\n";
    for (const auto& c : comments) header << "comment " << c.text << "\n";
    header << "element vertex " << count << "\n"
        << "property float x\n" << "property float y\n" << "property float z\n"
        << "property float f_dc_0\n" << "property float f_dc_1\n" << "property float f_dc_2\n"
        << "property float opacity\n"
        << "property float scale_0\n" << "property float scale_1\n" << "property float scale_2\n"
        << "property float rot_0\n" << "property float rot_1\n"
        << "property float rot_2\n" << "property float rot_3\n"
        << "end_header\n";
    return header.str();
}

std::string makeHeaderFullNoNormal(size_t count, const std::vector<PlyComment>& comments) {
    std::ostringstream header;
    header << "ply\n" << "format binary_little_endian 1.0\n";
    for (const auto& c : comments) header << "comment " << c.text << "\n";
    header << "element vertex " << count << "\n"
        << "property float x\n" << "property float y\n" << "property float z\n"
        << "property float f_dc_0\n" << "property float f_dc_1\n" << "property float f_dc_2\n";
    for (int i = 0; i < 45; ++i) header << "property float f_rest_" << i << "\n";
    header << "property float opacity\n"
        << "property float scale_0\n" << "property float scale_1\n" << "property float scale_2\n"
        << "property float rot_0\n" << "property float rot_1\n"
        << "property float rot_2\n" << "property float rot_3\n"
        << "end_header\n";
    return header.str();
}

std::string makeHeaderFull(size_t count, const std::vector<PlyComment>& comments) {
    std::ostringstream header;
    header << "ply\n" << "format binary_little_endian 1.0\n";
    for (const auto& c : comments) header << "comment " << c.text << "\n";
    header << "element vertex " << count << "\n"
        << "property float x\n" << "property float y\n" << "property float z\n"
        << "property float nx\n" << "property float ny\n" << "property float nz\n"
        << "property float f_dc_0\n" << "property float f_dc_1\n" << "property float f_dc_2\n";
    for (int i = 0; i < 45; ++i) header << "property float f_rest_" << i << "\n";
    header << "property float opacity\n"
        << "property float scale_0\n" << "property float scale_1\n" << "property float scale_2\n"
        << "property float rot_0\n" << "property float rot_1\n"
        << "property float rot_2\n" << "property float rot_3\n"
        << "end_header\n";
    return header.str();
}

/// ---------------------------------------- Writing ----------------------------------------

void GaussianLoader::writePlyFile(const std::string& filename,
    const std::vector<CPUGaussian>& gaussians,
    const std::vector<PlyComment>& comments)
{
    std::string path = resolveProjectPath(filename);
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << "\n";
        return;
    }

#ifdef DISABLE_SPHERICAL_HARMONICS
    std::string header = makeHeaderMinimal(gaussians.size(), comments);
#else

    std::string header = makeHeaderFull(gaussians.size(), comments);
#endif
    file.write(header.c_str(), header.size());

    const int rest_per_channel = SH_N - 1; // Usually 15 for degree 3

    for (const auto& g : gaussians) {
#ifdef DISABLE_SPHERICAL_HARMONICS
        PLYFileGaussianMinimal out{};

        out.f_dc[0] = g.sh[0];
        out.f_dc[1] = g.sh[1];
        out.f_dc[2] = g.sh[2];

        out.mean[0] = g.mean.x; out.mean[1] = g.mean.y; out.mean[2] = g.mean.z;
        out.opacity = std::log(g.opacity / (1.0f - g.opacity));
        glm::vec3 inv_scale = glm::log(g.scale);
        out.scale[0] = inv_scale.x; out.scale[1] = inv_scale.y; out.scale[2] = inv_scale.z;
        out.rotation[0] = g.rotation.x; out.rotation[1] = g.rotation.y;
        out.rotation[2] = g.rotation.z; out.rotation[3] = g.rotation.w;
        file.write(reinterpret_cast<const char*>(&out), sizeof(out));
#else
        PLYFileGaussianFull out{};
        out.mean[0] = g.mean.x; out.mean[1] = g.mean.y; out.mean[2] = g.mean.z;

        // f_dc
        out.sh[0] = g.sh[0];
        out.sh[1] = g.sh[1];
        out.sh[2] = g.sh[2];

        for (int j = 1; j < SH_N; ++j) {
            int rest_idx = j - 1;
            out.sh[3 + rest_idx] = g.sh[j * 3 + 0];                           // Red
            out.sh[3 + rest_idx + rest_per_channel] = g.sh[j * 3 + 1];        // Green
            out.sh[3 + rest_idx + 2 * rest_per_channel] = g.sh[j * 3 + 2];    // Blue
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
    std::cout << "Successfully wrote to file: " << path << "\n";
}
