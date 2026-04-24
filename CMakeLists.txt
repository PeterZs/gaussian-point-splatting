cmake_minimum_required(VERSION 3.27)

project(GaussianPointSplatting LANGUAGES C CXX CUDA)

# Require C++17 and CUDA17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/std:c++17>
    )
endif()

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)




# --- Collect your own sources recursively ---
file(GLOB_RECURSE SRC_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)
file(GLOB_RECURSE SRC_HEADERS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
)

# --- List package sources manually ---
set(PACKAGE_SOURCES
    packages/glad/src/glad.c
    packages/imgui/backends/imgui_impl_glfw.cpp
    packages/imgui/backends/imgui_impl_opengl3.cpp
    packages/imgui/imgui.cpp
    packages/imgui/imgui_demo.cpp
    packages/imgui/imgui_draw.cpp
    packages/imgui/imgui_tables.cpp
    packages/imgui/imgui_widgets.cpp
)

set(PACKAGE_HEADERS
    packages/glad/include/glad/glad.h
    packages/glad/include/KHR/khrplatform.h
    packages/imgui/backends/imgui_impl_glfw.h
    packages/imgui/backends/imgui_impl_opengl3.h
    packages/imgui/backends/imgui_impl_opengl3_loader.h
    packages/imgui/imgui.h
    packages/imgui/imgui_internal.h
    packages/imgui/imconfig.h
    packages/imgui/imstb_rectpack.h
    packages/imgui/imstb_textedit.h
    packages/imgui/imstb_truetype.h
    packages/nlohmann/json.hpp
    packages/nlohmann/json_fwd.hpp
    packages/stb/stb_image.h
    packages/stb/stb_image_write.h
    packages/tinybvh/tiny_bvh.h
)

# --- subdirectories ---
add_subdirectory(packages/glfw)



# --- Combine everything ---
add_executable(${PROJECT_NAME}
    ${SRC_SOURCES} ${SRC_HEADERS}
    ${PACKAGE_SOURCES} ${PACKAGE_HEADERS}
)



# CUDA settings
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    #INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "75;86;89")



set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# CUDA-specific compile options
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
#        -Xptxas -O3
#        -Xptxas -dlcm=ca
#        -Xptxas -dscm=ca
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -lineinfo
        -allow-unsupported-compiler
        -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
    >
)


# MSVC-specific options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            $<$<CONFIG:Debug>:/Od /RTC1>
            $<$<CONFIG:Release>:/O2 /Oi /Gy /RTC->
            
            /W3 /EHsc
        >
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:/INCREMENTAL>
        $<$<CONFIG:Release>:/OPT:REF /OPT:ICF>
        /DEBUG /SUBSYSTEM:CONSOLE
    )
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# Include dirs
target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/packages"
    "${CMAKE_CURRENT_SOURCE_DIR}/packages/glm"
    "${CMAKE_CURRENT_SOURCE_DIR}/packages/stb"
    "${CMAKE_CURRENT_SOURCE_DIR}/packages/imgui"
    "${CMAKE_CURRENT_SOURCE_DIR}/packages/glad/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/packages/tinybvh"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    CUDA::cublas
    glfw
)
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        opengl32
        kernel32 user32 gdi32 winspool comdlg32 advapi32 shell32 ole32 oleaut32 uuid odbc32 odbccp32
    )
else()
    find_package(OpenGL REQUIRED)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        OpenGL::GL
        Threads::Threads
        dl
    )
endif()