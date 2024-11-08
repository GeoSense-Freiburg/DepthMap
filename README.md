![DepthMap](DepthMap.png)

A Python package for depth map estimation using Depth-Anything. This tool processes images to generate depth maps using the Depth-Anything-V2 model, providing both visual depth maps and numerical depth data. The package automatically handles model downloading, environment setup, and provides an easy-to-use interface for batch processing images.

## Features
- Automatic download of pre-trained Depth-Anything-V2 models
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


Example with multiple parameters:

```bash
python -m DepthMap.main \
    input_folder=my_images \
    output.dir=results \
    depth_measurement=metric \
    dataset=hypersim \
    max_depth=10
```

### Output

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



