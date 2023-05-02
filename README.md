# Fast-Guided-Median-Filter

This repository contains the source code for the [Fast Guided Median Filter](https://ieeexplore.ieee.org/document/10007858) published in IEEE Transactions on Image Processing.

The paper proposes a guided filter kernel-based weighted median filter that is compatible with both CPU and GPU, and capable of processing multidimensional, multichannel, and high precision data at high speed.

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
- The code is in folder FGMF_CPU
### List-O(1)
- O(1) sliding window on CPU for high precision data
- The code can handle 2D grayscale and color images
- The code is in folder FGMF_List
### GPU-O(r)
- GPU implementation with O(r) sliding window approach
- The code can handle 2D grayscale, color, and multichannel images, and multidimensional grayscale/color data.
- The code is in folder FGMF_GPU

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
FGMF_CPU.exe -i <input_image_path> [-g <guide_image_path>] [-r <radius>] [-e <root_epsilon>] [-t <number_of_threads>] [-s <output_image_path>]

```

### Options

| Option | Description |
| --- | --- |
| -i, --input | (Required) Input image path. |
| -g, --guide | (Optional) Guide image path. Default: same as input image path. |
| -r, --radius | (Optional) Window radius (int). Default: 5. |
| -e, --root-epsilon | (Optional) Root of epsilon (float). Default: 2.55. |
| -t, --threads | (Optional) Number of threads (int). Default: number of available processors. |
| -s, --save | (Optional) Output image save path. Default: output.png in the same folder as the program. |


**Note:**

- The **`-e`** option represents the root of the parameter $\epsilon$ in the paper, and it is squared internally in the program. For example, **`-e 25.5`** means $\epsilon = 25.5^2$.
- The default number of threads is set to the number of available processors using the **`omp_get_num_procs()`** function.

### Example

```
FGMF_CPU.exe -i input.png -g guide.png -r 10 -e 25.5 -t 4 -s output.png

```


## Usage for List-O(1)

```
FGMF_List.exe -i <input_image_path> [-g <guide_image_path>] [-r <radius>] [-e <root_epsilon>] [-t <number_of_threads>] [-s <output_image_path>]

```

### Options

Same as CPU-O(1).


### Example

```
FGMF_List.exe -i input.png -g guide.png -r 10 -e 25.5 -t 4 -s output.png

```


## Usage for GPU-O(r) for 2D grayscale/color images

```
FGMF_GPU.exe -i <input_image_path> [-g <guide_image_path>] [-r <radius>] [-e <root_epsilon>] [-s <output_image_path>] [-b <block_size>]

```

### Options

| Option | Description |
| --- | --- |
| -i, --input | (Required) Input image path. |
| -g, --guide | (Optional) Guide image path. Default: same as input image path. |
| -r, --radius | (Optional) Window radius (int). Default: 5. |
| -e, --root-epsilon | (Optional) Root of epsilon (float). Default: 2.55. |
| -s, --save | (Optional) Output image save path. Default: output.png in the same folder as the program. |
| -b, --block-size | (Optional) Block size for CUDA thread (int). Default: 16. |

**Note:**

- The **`-e`** option represents the root of the parameter $\epsilon$ in the paper, and it is squared internally in the program. For example, **`-e 25.5`** means $\epsilon = 25.5^2$.
- The **`-b`** option sets the block size for CUDA threads. Adjust this according to your GPU capabilities. Note that if you encounter the error message "GPUassert: an illegal memory access was encountered" during execution, this may indicate that the block size is too large.

### Example

```
FGMF_GPU.exe -i input.png -g guide.png -r 10 -e 25.5 -s output.png -b 32

```


## Usage for GPU-O(r) for 2D multichannel images

```
FGMF_GPU_MC.exe -i <input_image_path> [-d <sequence_digit_count>] [-n <sequence_start_number>] [-u <use_data_count>] [-x <output_image_extension>] [-g <guide_image_path>] [-r <radius>] [-c <channel_radius>] [-a] [-e <root_epsilon>] [-s <output_image_path>] [-b <block_size>]

```

### Options

| Option | Description |
| --- | --- |
| -i, --input | (Required) Input image path without the numeric sequence. |
| -d, --digit-count | (Optional) Minimum number of digits in a sequence, padded with zeros if necessary. Default: 3. |
| -n, --start-number | (Optional) Starting number in the sequence. Default: 0. |
| -u, --use-data-count | (Optional) Number of images to use. Default: 3. |
| -x, --extension | (Optional) Input and output image extension. Default: .png. |
| -g, --guide | (Optional) Guide image path without the numeric sequence. Default: same as input image path. |
| -r, --radius | (Optional) Spatial radius (int). Default: 5. |
| -c, --channel-radius | (Optional) Channel radius (int). Set to -1 to use all channels of the guide image for filtering. Default: 2. |
| -a, --avoid-same-channel | (Optional) Avoid using the same channel as a guide. Default: false. |
| -e, --root-epsilon | (Optional) Root of epsilon (float). Default: 2.55. |
| -s, --save | (Optional) Output image save path without the numeric sequence. Default: outputXXX.png in the same folder as the program. |
| -b, --block-size | (Optional) Block size for CUDA thread (int). Default: 16. |

**Note:**

- The **`-e`** option represents the root of the parameter $\epsilon$ in the paper, and it is squared internally in the program. For example, **`-e 25.5`** means $\epsilon = 25.5^2$.
- The **`-b`** option sets the block size for CUDA threads. Adjust this according to your GPU capabilities. Note that if you encounter the error message "GPUassert: an illegal memory access was encountered" during execution, this may indicate that the block size is too large.
- The **`-a`** option can be used to prevent using the same channel as a guide when filtering. This setup is used in the experiment in Sec. VII-E in the paper.


### How to handle multichannel data
The multichannel data you wish to process should be saved as separate single-channel images, with each channel stored in a separate image file. These files should include a sequential number after the name. For instance, when dealing with 4 channels of data, each channel should be individually saved prior to processing with names like "input00.png", "input01.png", "input02.png", "input03.png", and so on, where the number indicates the channel sequence.

### Example

```
FGMF_GPU_MC.exe -i input -d 4 -n 1 -u 5 -x .png -g guide -r 10 -c 3 -a -e 25.5 -s output -b 16

```

In this example, the input image path sequence is interpreted as "input0001.png", "input0002.png", "input0003.png", "input0004.png", and "input0005.png". 
Similarly, the guide image path sequence is interpreted as "guide0001.png", "guide0002.png", and so on, up to "guide0005.png".
The output is written to "output0001.png", "output0002.png", etc.



## Usage for GPU-O(r) for multidimensional grayscale/color data

```
FGMF_GPU_MD.exe -i <input_image_path> [-d <sequence_digit_count>] [-n <sequence_start_number>] [-u <use_data_count>] [-x <output_image_extension>] [-g <guide_image_path>] [-r <radius>] [--radii-3d <radius_1,radius_2,...>] [--size-3d <size_1,size_2,...>] [-e <root_epsilon>] [-s <output_image_path>] [-b <block_size>]
```

### Options

| Option | Description |
| --- | --- |
| -i, --input | (Required) Input image path without the numeric sequence. |
| -d, --digit-count | (Optional) Minimum number of digits in a sequence, padded with zeros if necessary. Default: 3. |
| -n, --start-number | (Optional) Starting number in the sequence. Default: 0. |
| -u, --use-data-count | (Optional) Number of images to use. Default: 3. |
| -x, --extension | (Optional) Output image extension. Default: .png. |
| -g, --guide | (Optional) Guide image path without the numeric sequence. Default: same as input image path. |
| -r, --radius | (Optional) Spatial radius (int). Default: 5. |
| --radii-3d | (Optional) Comma-separated list of filter radii for 3rd dimension and beyond. Default: 0. |
| --size-3d | (Optional) Comma-separated list of sizes for 3rd dimension and beyond. Default: 1. |
| -e, --root-epsilon | (Optional) Root of epsilon (float). Default: 2.55. |
| -s, --save | (Optional) Output image save path without the numeric sequence. Default: outputXXX.png in the same folder as the program. |
| -b, --block-size | (Optional) Block size for CUDA thread (int). Default: 16. |

**Note:**

- The **`-e`** option represents the root of the parameter $\epsilon$ in the paper, and it is squared internally in the program. For example, **`-e 25.5`** means $\epsilon = 25.5^2$.
- The **`-b`** option sets the block size for CUDA threads. Adjust this according to your GPU capabilities. Note that if you encounter the error message "GPUassert: an illegal memory access was encountered" during execution, this may indicate that the block size is too large.
- The **`--radii-3d`** and **`--size-3d`** options allow for setting filter radii and sizes for the third dimension and beyond.


### How to handle multidimensional data
When handling N-dimensional data, it's necessary to divide the data into multiple 2D image data. For instance, if you want to use 5-dimensional data, you should arrange 2D image data in a 3D manner. 

Take, for example, a 5-dimensional data set with a size of 100x100x4x3x2. The first two dimensions are treated as image data. Let's say you want the filter radius to be 5x5x3x2x1. In this case, you need to decompose the 5-dimensional data into 24 pieces of image data, each with a sequential name.

The arrangement of the data can be visualized as follows:

```
| |-------------->3rd
| |00, 01, 02, 03
| |04, 05, 06, 07
| |08, 09, 10, 11
| 4th
| 
| |-------------->3rd
| |12, 13, 14, 15
| |16, 17, 18, 19
| |20, 21, 22, 23
| 4th
5th 
```

To execute the above settings, set the options as follows:

```
-r 5 --radii-3d 3,2,1 --size-3d 4,3,2 -d 2 -n 0 -u 24
```


### Example

```
FGMF_GPU_MD.exe -i input -d 4 -n 1 -u 24 -x .png -g guide -r 10 --radii-3d 1,2,3 --size-3d 2,3,4 -e 25.5 -s output -b 16
```

The product of the numbers specified with the --size-3d option must match the number specified with the -u option.




## Notes
- While the code for multidimensional data is designed to handle data of 5 dimensions or more, please note that I have only verified its functionality with up to 4-dimensional data.
- This code has been refactored to improve readability. As a result, the speed has decreased by approximately 1.2 times compared to the speed reported in the paper.
- The computation times reported in the paper does not include uploading and downloading images to/from GPU memory.
- Proper processing speed measurements require a "warm-up" run at the start of the measurement that is not included in the measurement.