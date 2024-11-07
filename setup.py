from setuptools import setup, find_packages
import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import logging
from tqdm import tqdm

def setup_logging():
    """Configure logging with custom formatter."""
    formatter = logging.Formatter('%(asctime)s - Module(%(module)s):Line(%(lineno)d) %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_model_weights():
    """Download the model weights if they don't exist."""
    models = {
        "depth_anything_v2_metric_hypersim_vitl.pth": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true",
        "depth_anything_v2_metric_vkitti_vitl.pth": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true"
    }
    
    model_dir = Path("models/test/Depth-Anything-V2/metric_depth/checkpoints")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, url in models.items():
        target_path = model_dir / model_name
        if not target_path.exists():
            logging.info(f"Downloading {model_name} from {url}")
            try:
                download_url(url, target_path)
                logging.info(f"Successfully downloaded {model_name}")
            except Exception as e:
                logging.error(f"Failed to download {model_name}: {e}")
                sys.exit(1)
        else:
            logging.info(f"{model_name} already exists")

def clone_depth_anything():
    """Clone the Depth-Anything repository if it doesn't exist."""
    repo_path = Path("models/test/Depth-Anything-V2")
    if not repo_path.exists():
        logging.info("Cloning Depth-Anything repository...")
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/DepthAnything/Depth-Anything-V2",
                str(repo_path)
            ], check=True)
            logging.info("Successfully cloned Depth-Anything repository")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to clone repository: {e}")
            sys.exit(1)
    else:
        logging.info("Depth-Anything repository already exists")

def check_conda_env(env_name):
    """Check if conda environment exists."""
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        return env_name in result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to check conda environments: {e}")
        return False

def setup_conda_environment():
    """Create conda environment from environment.yaml if it doesn't exist."""
    env_name = "depthmap"
    yaml_path = Path("environment.yaml")
    
    if not yaml_path.exists():
        logging.error("environment.yaml not found")
        sys.exit(1)
        
    if not check_conda_env(env_name):
        logging.info(f"Creating conda environment '{env_name}'...")
        try:
            subprocess.run(['conda', 'env', 'create', '-f', str(yaml_path)], check=True)
            logging.info(f"Successfully created conda environment '{env_name}'")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create conda environment: {e}")
            logging.error(f"Please check the environment.yaml file and try to install it manually (with >>> conda env create -f environment.yaml <<< in the terminal)")
            sys.exit(1)
    else:
        logging.info(f"Conda environment '{env_name}' already exists")

def main():
    """Main function to setup the environment without package installation."""
    setup_logging()
    setup_conda_environment()
    clone_depth_anything()
    download_model_weights()

def build_package():
    """Function to build and install the package."""
    setup(
        name="depthmap",
        version="0.1.0",
        packages=find_packages(),
        author="Luis Kremer",
        author_email="luis.kremer@geosense.uni-freiburg.de",
        description="A depth map estimation package using Depth-Anything",
        long_description=open('README.md').read(),
        long_description_content_type="text/markdown",
        python_requires=">=3.8",
        entry_points={
            'console_scripts': [
                'depthmap=depthmap.main:main',
            ],
        },
    )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If arguments are provided, run setuptools
        build_package()
    else:
        # Just setup the environment
        main()