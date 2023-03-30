# Fast-Guided-Median-Filter

This repository contains the source code for the [Fast Guided Median Filter](https://ieeexplore.ieee.org/document/10007858) published in IEEE Transactions on Image Processing.
The code is released under the MIT License.
Please cite our paper when you use or modify this code.

K. Mishiba, "Fast Guided Median Filter," in IEEE Transactions on Image Processing, vol. 32, pp. 737-749, 2023, doi: 10.1109/TIP.2022.3232916.

```
@ARTICLE{Mishiba2023,
  author={Mishiba, Kazu},
  journal={IEEE Transactions on Image Processing}, 
  title={Fast Guided Median Filter}, 
  year={2023},
  volume={32},
  pages={737-749},
  doi={10.1109/TIP.2022.3232916}}

```

## Implementation
### CPU-O(1)
- CPU implementation with O(1) sliding window approach
- The code can handle 2D grayscale and color images
- The code is available in this repository
### GPU-O(r)
- GPU implementation with O(r) sliding window approach
- The code, which can handle 2D grayscale, color, multispectral images, 3D and 4D grayscale/color images, will be released soon
### List-O(1)
- O(1) sliding window on CPU for high precision data
- The code, which can handle 2D grayscale and color images, will be released soon

## System Requirements
The code has been tested in the following environment:
- Windows 11
- Visual Studio Community 2022
- Intel CPU with AVX2 support
- NVIDIA Graphics card with CUDA support
- CUDA 11.7
- OpenCV 4 installed using vcpkg
- OpenMP
- C++ 14


## Usage for CPU-O(1)

```
FGMF_CPU.exe -i <input_image_path> [-g <guide_image_path>] [-r <radius>] [-e <root_epsilon>] [-b <bit_depth>] [-t <number_of_threads>] [-s <output_image_path>]

```

### Options

| Option | Description |
| --- | --- |
| -i, --input | (Required) Input image path. |
| -g, --guide | (Optional) Guide image path. Default: same as input image path. |
| -r, --radius | (Optional) Window radius (int). Default: 5. |
| -e, --root-epsilon | (Optional) Root of epsilon (float). Default: 2.55. |
| -b, --bit-depth | (Optional) Bit depth per channel (int). Default: 8. |
| -t, --threads | (Optional) Number of threads (int). Default: number of available processors. |
| -s, --save | (Optional) Output image save path. Default: output.png in the same folder as the program. |


**Note:**

- The **`e`** option represents the root of the parameter $\epsilon$ in the paper, and it is squared internally in the program. For example, **`e 25.5`** means $\epsilon = 25.5^2$.
- The default number of threads is set to the number of available processors using the **`omp_get_num_procs()`** function.

### Example

```
FGMF_CPU.exe -i input.png -g guide.png -r 10 -e 25.5 -b 8 -t 4 -s output.png

```


## Usage for GPU-O(r)
Available soon.

## Usage for List-O(1)
Available soon.


## Notes
- This code has been refactored to improve readability. As a result, the speed has decreased by approximately 1.2 times compared to the speed reported in the paper.
- The computation times reported in the paper does not include uploading and downloading images to/from GPU memory.
- Proper processing speed measurements require a "warm-up" run at the start of the measurement that is not included in the measurement.