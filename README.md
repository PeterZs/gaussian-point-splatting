# Gaussian Point Splatting

![Teaser image showing over 4 copies of a town rendered using Gaussians, with two magnified insets.](teaser.png?raw=true)
<p align="center">
  <em>Rendering over 425M Gaussians interactively on an RTX 4070 Ti SUPER. We achieve this high throughput with a stochastic
approach (the images shown here are converged) but avoid approximations, level-of-detail mechanisms and complex data structures</em>
</p>

We propose Gaussian point splatting, a stochastic method for rendering massive [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) scenes. By sampling pixel-sized opaque points, splatting them atomically, and applying stochastic transparency, we eliminate the need for sorting. Our approach distributes workload evenly across GPU threads, enabling the real-time display of hundreds of millions of Gaussians efficiently.




## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{rijsdijk2026gps,
  title={Gaussian Point Splatting},
  author={Rijsdijk, J. and Peters, C. and Marroquim, R. and Weinnman, M.},
  journal={ACM Trans. Graph.},
  publisher={Association for Computing Machinery},
  year={2026}
}
```

## How to Run

The codebase uses a command-line interface for configuring rendering settings. Below are the available arguments to control the application behavior.

### Example Usage

```bash
./gaussian_splatting --model-path ./data/scene_01 --resolution 1920 1080 --samples-per-pixel 4
```

### Command Line Arguments

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--model-path` | `string` | Path to the trained Gaussian point model. (Either a .ply file, or the root folder of a [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) output.) |
| `--resolution` | `int, int` | Sets the render window resolution (e.g., `--resolution 1920 1080`). |
| `--samples-per-pixel` | `int` | Sets the number of samples per pixel. |
| `--background` | `float, float, float` | Sets the RGB background color. |
| `--fovy` | `float` | Overrides the default camera Field of View. |
| `--camera-path` | `string` | Load a pre-defined camera trajectory, expects a json file exported by [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). |
| `--play-camera-path` | Flag | Automatically play the loaded camera path on start. |
| `--disable-gui` | Flag | Disables the UI overlay. |
| `--enable_profiling` | Flag | Enables [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) profiling hooks. |

### Culling & Optimization Flags

| Argument | Description |
| :--- | :--- |
| `--cull-small` | Enables culling for small Gaussian splats from [Splatshop](https://github.com/m-schuetz/splatshop). |
| `--disable-hierarchical-culling` | Disables hierarchical culling optimizations. |
| `--disable-occlusion-culling` | Disables occlusion culling. |
| `--sort-morton-order` | Enables Morton order sorting for faster processing and lower (system) RAM usage. |
| `--reduce-point-count` | Uses K = (super sampling rate)^2^ as described in the paper. |

-----

## Important Files

This section highlights the core components of the codebase where the primary implementation resides.

| File | Description |
| :--- | :--- |
| `main.cu`| Render loop and program entry |
| `src/config.h` | Some compile-time constants, for instance to toggle spherical harmonics and compression. |
| `src/core/rendering/passes/gaussian_point_splatting.cu` | Implementation of the main method, including kernels for preprocessing, splatting, and combining of samples. |
| `src/core/rendering/passes/post_processing_kernels.cu` | Implementation of temporal reuse. |
| `src/core/random/workload/workload_distributor.cuh` | Handles the workload distribution algorithm as discussed in the paper. |
| `src/core/rendering/occlusion/gaussian_bvh.cuh` | Computes the hierarchy and handles hierarchical culling. |
| `src/core/rendering/occlusion/depth_mip_chain.cuh` | Computes the depth mip chain and exposes a device function to check whether an AABB at a certain depth is occluded or not. |

-----

## Prerequisites

  * CUDA Toolkit (12.6 recommended)
  * CMake (\>= 3.27)
  * All others are in `/packages`

## License
