from setuptools import setup, find_packages
import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import logging
from tqdm import tqdm
import platform

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

def download_relative_model_weights():
    """Download the model weights (relative values as output) if they don't exist."""
    models = {
        "depth_anything_v2_vitl.pth": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
    }
    
    model_dir = Path("models/Depth-Anything-V2/checkpoints")
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

def download_metric_model_weights():
    """Download the model weights (metric output) if they don't exist."""
    models = {
        "depth_anything_v2_metric_hypersim_vitl.pth": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true",
        "depth_anything_v2_metric_vkitti_vitl.pth": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true"
    }
    
    model_dir = Path("models/Depth-Anything-V2/metric_depth/checkpoints")
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
    repo_path = Path("models/Depth-Anything-V2")
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

def get_conda_lockfile():
    """Get the appropriate conda lock file based on OS."""
    system = platform.system().lower()
    if system == 'linux':
        return 'conda-linux-64.lock'
    elif system == 'darwin':
        return 'conda-osx-64.lock'
    elif system == 'windows':
        return 'conda-win-64.lock'
    else:
        raise OSError(f"Unsupported operating system: {system}")

def setup_conda_environment():
    """Create conda environment from appropriate lock file if it doesn't exist."""
    env_name = "depthmap"
    lock_file = get_conda_lockfile()
    logging.info(f"Using conda lock file: {lock_file}")
    lock_path = Path(lock_file)
    
    if not lock_path.exists():
        logging.error(f"{lock_file} not found")
        sys.exit(1)
        
    if not check_conda_env(env_name):
        logging.info(f"Creating conda environment '{env_name}'...")
        try:
            subprocess.run(['conda', 'create', '--name', env_name, '--file', str(lock_path)], check=True)
            logging.info(f"Successfully created conda environment '{env_name}'")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create conda environment: {e}")
            logging.error(f"Please check the {lock_file} file and try to install it manually (with >>> conda create --name {env_name} --file {lock_file} <<< in the terminal)")
            sys.exit(1)
    else:
        logging.info(f"Conda environment '{env_name}' already exists")

def main():
    """Main function to setup the environment without package installation."""
    setup_logging()
    setup_conda_environment()
    clone_depth_anything()
    download_relative_model_weights()
    download_metric_model_weights()

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
