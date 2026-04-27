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
@article{rijsdijk2026gps,
  title={Gaussian Point Splatting},
  author={Rijsdijk, Joris and Peters, Christoph and Marroquim, Ricardo and Weinnman, Michael},
  journal={ACM Trans. Graph.},
  volume = {45},
  number = {4},
  publisher={Association for Computing Machinery},
  year={2026}
}
```


## How to Run

### Prerequisites

  * CUDA Toolkit (developed with version 12.6)
  * CMake (\>= 3.27)
  * All others are included directly in `/packages`

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
| `--samples-per-pixel <n>` | Sets the number of samples per pixel. |
| `--background <r> <g> <b>` | Sets the RGB background color. |
| `--fovy <degrees>` | Overrides the default camera Field of View. |
| `--camera-path <path>` | Load a `.json` camera trajectory exported by 3DGS. If `--model--path` points to a 3DGS output directory, it automatically loads the camera path. |
| `--play-camera-path` | Automatically play the loaded camera path on start. |
| `--disable-gui` | Disables the UI overlay. |
| `--cull-small` | Enables culling for small Gaussian splats from [Splatshop](https://github.com/m-schuetz/splatshop). |
| `--disable-hierarchical-culling` | Disables hierarchical culling optimizations. |
| `--disable-occlusion-culling` | Disables occlusion culling. |
| `--sort-morton-order` | Enables Morton order sorting for faster processing and lower (system) RAM usage. |
| `--reduce-point-count` | Uses K = (super sampling rate)^2 as described in the paper. |

-----

### Important Files

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

## Data Credits

The large-scale scenes used in our teaser and evaluations were provided by **Andrii Shramko**.

> "3D scanning data created and provided by Andrii Shramko, Teleportour. [https://www.linkedin.com/in/andrii-shramko/](https://www.linkedin.com/in/andrii-shramko/) 
> [https://www.linkedin.com/company/teleportour/](https://www.linkedin.com/company/teleportour/) 
> [teleportour.com](http://teleportour.com)"
