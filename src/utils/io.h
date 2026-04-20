#pragma once
#include <string>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cerrno>
#include <cstring>  // for strerror

#ifdef _WIN32
#include <direct.h>   // for _mkdir
#include <sys/stat.h>
#include <sys/types.h>
#else
#include <sys/stat.h> // for mkdir, stat
#include <sys/types.h>
#endif

#include <filesystem>
namespace fs = std::filesystem;

// Helper function to check if a directory exists.
bool inline directory_exists(const std::string& path) {
#ifdef _WIN32
    struct _stat info;
    if (_stat(path.c_str(), &info) != 0)
        return false;
    return (info.st_mode & S_IFDIR) != 0;
#else
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
        return false;
    return (info.st_mode & S_IFDIR) != 0;
#endif
}

// Recursively creates the directory and any required parent directories.
bool inline create_directory(const std::string& dir_path) {
    if (dir_path.empty()) {
        return false;
    }

    if (directory_exists(dir_path)) {
        return true;
    }

    // Recursively create the parent directory if it doesn't exist.
    size_t pos = dir_path.find_last_of("/\\");
    if (pos != std::string::npos) {
        std::string parent = dir_path.substr(0, pos);
        if (!parent.empty() && !directory_exists(parent)) {
            if (!create_directory(parent)) {
                return false;
            }
        }
    }

    // Create the target directory.
#ifdef _WIN32
    int ret = _mkdir(dir_path.c_str());
#else
    int ret = mkdir(dir_path.c_str(), 0755);
#endif

    if (ret != 0 && errno != EEXIST) {
        std::cerr << "Failed to create directory \"" << dir_path << "\": " << strerror(errno) << std::endl;
        return false;
    }

    return true;
}

inline std::string getCurrentDateTime() {
    auto now = std::chrono::system_clock::now();

    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm local_tm;
#if defined(_MSC_VER)
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y-%m-%d,%H-%M-%S");
    return oss.str();
}

// Converts a relative path into an absolute path relative to PROJECT_ROOT
inline std::string resolveProjectPath(const std::string& relativePath) {
    fs::path root(PROJECT_ROOT);   // injected by CMake
    fs::path fullPath = root / relativePath;
    return fs::absolute(fullPath).u8string();
}
