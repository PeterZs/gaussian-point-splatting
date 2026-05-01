# Gaussian Point Splatting

![Teaser image showing over 4 copies of a town rendered using Gaussians, with two magnified insets.](teaser.png?raw=true)
<p align="center">
  <em>Rendering over 425M Gaussians interactively on an RTX 4070 Ti SUPER. We achieve this high throughput with a stochastic
approach (the images shown here are converged) but avoid approximations, level-of-detail mechanisms and complex data structures.</em>
</p>

We propose Gaussian point splatting, a stochastic method for rendering massive [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) scenes. By sampling pixel-sized opaque points, splatting them atomically, and applying stochastic transparency, we eliminate the need for sorting. Our approach distributes workload evenly across GPU threads, enabling the real-time display of hundreds of millions of Gaussians efficiently.



## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{Rijsdijk2026GaussianPointSplatting,
  title = {Gaussian Point Splatting},
  author = {Rijsdijk, Joris and Peters, Christoph and Marroquim, Ricardo and Weinnman, Michael},
  journal = {ACM Trans. Graph.},
  volume = {45},
  number = {4},
  publisher = {Association for Computing Machinery},
  year = {2026},
  doi = {10.1145/3811272}
}
```


## How to Run

### Prerequisites

  * CUDA Toolkit (developed with version 12.6)
  * CMake (\>= 3.27)
  * All other third-party packages are included directly in `/packages`
  * **Note:** This application requires an NVIDIA GPU to work. On systems with multiple GPUs (e.g., laptops with integrated graphics), you must ensure the application is configured to run on your dedicated NVIDIA GPU. If the executable defaults to the integrated GPU, the CUDA-OpenGL interoperability will fail to initialize.

### Installation

 * Clone the repository 
 * Build the project using CMake


### Example Usage

```bash
./gaussian_splatting --model-path ./data/scene_01 --resolution 1920 1080 --samples-per-pixel 4
```

### Command Line Arguments

| Argument | Description |
| :--- | :--- |
| `--model-path <path>` | Path to the trained Gaussian point model. (Either a `.ply` file, or the root folder of a 3DGS output.) |
| `--resolution <w> <h>` | Sets the render window resolution (e.g., `--resolution 1920 1080`). |
| `--supersampling-factor <n>` | Sets the supersampling factor, default 2. (i.e. 2x2 samples per output pixel) |
| `--samples-per-pixel <n>` | Sets the number of samples per pixel, default 1. |
| `--fovy <degrees>` | Overrides the default camera Field of View. |
| `--camera-path <path>` | Load a `.json` camera trajectory exported by 3DGS. If `--model--path` points to a 3DGS output directory, it automatically loads the camera path. |
| `--disable-hierarchical-culling` | Disables hierarchical culling optimizations. |
| `--disable-occlusion-culling` | Disables occlusion culling. |
| `--sort-morton-order` | Enables Morton order sorting for faster processing and lower (system) RAM usage. |
| `--one-point-per-thread` | If enabled, the renderer uses K = 1 (as described in the paper). By default, K = (supersampling factor)^2. Enabling this flag enables mathematically more correct rendering, at the cost of (much) higher frame times. |
| `--play-camera-path` | Automatically play the loaded camera path on start. |
| `--disable-gui` | Disables the UI overlay. |
| `--cull-small` | Enables culling for small Gaussian splats from [Splatshop](https://github.com/m-schuetz/splatshop). This does not make a big difference in frame time for our renderer. |
| `--background <r> <g> <b>` | Sets the RGB background color. |


### Controls

- Use WASD (horizontal movement), space bar (vertical up), ctrl (vertical down) and the mouse (rotation) to control the camera.
- Use backspace to toggle the visibility of the UI.
- Use escape to toggle mouse pointer lock.
- Use F5 to take a screenshot, saved at `./out/screenshots`.
-----

## Important Files

| File | Description |
| :--- | :--- |
| [`src/main.cu`](src/main.cu) | Render loop and program entry |
| [`src/config.h`](src/config.h) | Defines compile-time constants, for instance to toggle spherical harmonics and compression. |
| [`src/core/rendering/passes/gaussian_point_splatting.cu`](src/core/rendering/passes/gaussian_point_splatting.cu) | Implementation of the main method, including kernels for preprocessing, splatting, and combining of samples. |
| [`src/core/rendering/passes/post_processing_kernels.cuh`](src/core/rendering/passes/post_processing_kernels.cuh) | Implementation of temporal reuse. |
| [`src/core/random/workload/workload_distributor.cuh`](src/core/random/workload/workload_distributor.cuh) | Handles the workload distribution algorithm as discussed in the paper. |
| [`src/core/rendering/occlusion/gaussian_bvh.cu`](src/core/rendering/occlusion/gaussian_bvh.cu) | Computes the hierarchy and handles hierarchical culling. |
| [`src/core/rendering/occlusion/depth_mip_chain.cu`](src/core/rendering/occlusion/depth_mip_chain.cu) | Computes the depth mip chain and exposes a device function to check whether an AABB at a certain depth is occluded or not. |

-----


## License

This project is licensed under the **BSD 3-Clause License** - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the Dutch Research Council (NWO) under the project *VR Retrofit-4U* (Grant ID: [10.61686/EIHMV70145](https://doi.org/10.61686/EIHMV70145)).

The large-scale scenes used in our teaser and evaluations were provided by [Andrii Shramko](https://www.linkedin.com/in/andrii-shramko/), [Teleportour](https://teleportour.com/).

We also thank the authors of the following works for the remaining scenes:
- [Tanks and Temples](https://tanksandtemples.org/)
- [Mip-NeRF](https://jonbarron.info/mipnerf/)
- [Deep Blending](https://arxiv.org/abs/1808.06579)