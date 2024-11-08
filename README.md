![DepthMap](DepthMap.png)

A Python workflow for depth map estimation using ![Depth-Anything](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file). This tool processes images to generate depth maps using the Depth-Anything-V2 model, providing both visual depth maps and numerical depth data. The package automatically handles model downloading, environment setup, and provides an easy-to-use interface for batch processing images.

## Features
- Automatic download of pre-trained Depth-Anything-V2 models (large version only)
- Batch processing of images
- Outputs both visual depth maps and raw depth data
- Statistical analysis of depth values
- CSV export of depth statistics

## Installation

Make sure you have Conda installed. If not, download it from [Miniconda](https://docs.anaconda.com/miniconda/)

To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Luis-Kr/DepthMap.git

# Navigate into the directory
cd DepthMap

# Run setup script (this will download large models >2GB)
python setup.py
```

## Usage

Run the depth estimation with default settings:

```bash
# Activate the conda environment
conda activate depthmap

# Run the main script to create the depth maps
python -m DepthMap.main input_folder=path/to/images output.dir=path/to/output
```

### Configuration

The default configuration can be found in config/main.yaml. You can override any of these settings via command line:

- `input_folder`: Directory containing input images
- `output.dir`: Directory for saving depth maps
- `depth_measurement`: Choose between "relative" or "metric" depth
  - `relative`: Provides relative depth values (better for general use)
  - `metric`: Fine-tuned version that attempts to predict actual depth in meters
- `encoder`: Model encoder type (currently only "vitl" supported)
- `dataset`: For metric depth, choose between:
  - `hypersim`: Optimized for indoor scenes
  - `vkitti`: Optimized for outdoor scenes
- `max_depth`: Maximum depth (in meters) to be expected in the input image (only for metric depth)


Example with multiple parameters:

```bash
python -m DepthMap.main \
    input_folder=my_images \
    output.dir=results \
    depth_measurement=metric \
    dataset=hypersim \
    max_depth=10
```

## Output

The tool generates:

1. Colored depth maps in `<output.dir>/images/`
2. Raw depth arrays in `<output.dir>/arrays/`
3. CSV file with depth statistics including:
    - 5th percentile depth
    - 95th percentile depth
    - Mean depth
    - Median depth

## Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda package manager

## License

This project has two licensing components:

1. **Depth-Anything Model**: The Depth-Anything model is governed by its own license terms. You must comply with the [Depth-Anything License](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE) for any use of the model.

2. **This Project**: The DepthMap workflow code (excluding the Depth-Anything model) is released under the MIT License. See the [LICENSE](LICENSE) file for details.


