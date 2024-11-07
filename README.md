# DepthMap

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
git clone https://github.com/username/DepthMap.git

# Navigate into the directory
cd DepthMap

# Run setup script (this will download large models >2GB)
python setup.py
```
