#pragma once

// ------------------------- Stochastic Gaussian Splatting -------------------------

// variables

#define GAUSSIAN_TRUNCATION_STD 3.0f
#define POINTS_PER_KERNEL 1
#define TARGET_NUM_HIERARCHICAL_NODES 1e6 // Number of groups used to in a rough culling pass.

// binary toggles

#define USE_COMPRESSED_GAUSSIANS // If defined, use 16 bytes for geometry, 60 bytes for SH (if enabled).
//#define DISABLE_SPHERICAL_HARMONICS // If defined, SH evaluation is disabled. This significantly decreases the VRAM footprint of the renderer.
//#define USE_SDR_COLOR // If defined, uses a 32 bit word instead of a vec3 to store color per Gaussian, DISABLE_SPHERICAL_HARMONICS preferably used together with.
//#define ENABLE_FREEZING_CULLING // If defined, an option appears in the menu to freeze the frustum so occlusion culling can be inspected. Used in Fig 13. in the paper.
//#define ENABLE_IMMEDIATE_SPLATTING //If defined, an option appears in the menu to splat points during the preprocessing step, may be slower in many cases.

// ------------------------- Constants (do not touch!) -------------------------

#ifndef GAUSSIAN_OPACITY_TYPE
#define GAUSSIAN_OPACITY_TYPE uint8_t 
#endif

#ifndef WEIGHT_MASK_TYPE
#define WEIGHT_MASK_TYPE uint32_t // Store 32 flags/word: a vector is length (len(Gaussians) / 32)
#endif

#ifndef WEIGHT_TYPE
#define WEIGHT_TYPE float
#endif

//64 bit interleaved (depth - color) definition:
#define COLOR_BITS 36ull
#define DEPTH_BITS 28ull
#define DEPTH_MAX ((1ull << DEPTH_BITS) - 1)
#define DEPTH_SHIFT COLOR_BITS
#define COLOR_MASK ((1ull << COLOR_BITS) - 1)
#define DEPTH_MASK (DEPTH_MAX << DEPTH_SHIFT)


#define MORTON_ORDER_IMAGE_BUFFER
#define MORTON_ORDER_TILE_SIZE 64
#define MORTON_ORDER_TILE_BITS 6

#define MAX_GAUSSIAN_OPACITY 1.0f