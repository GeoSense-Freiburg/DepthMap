import sys
import torch
import logging
import warnings
import cv2
import numpy as np
import csv
from pathlib import Path
from typing import Any, Optional, Tuple, List, Set, Union
from dataclasses import dataclass

# Hydra and OmegaConf imports
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

@dataclass
class Config:
    input_folder: str
    output_dir: str
    depth_measurement: str
    encoder: str
    dataset: str
    max_depth: int

def validate_config(cfg: DictConfig) -> Config:
    required_fields = {
        'input_folder': str,
        'output.dir': str,
        'depth_measurement': str,
        'encoder': str,
        'dataset': str,
        'max_depth': int
    }
    
    for field, expected_type in required_fields.items():
        value = cfg.get(field) if '.' not in field else cfg.output.get(field.split('.')[1])
        if value is None:
            raise ValueError(f"Missing required config field: {field}")
        if not isinstance(value, expected_type):
            raise TypeError(f"Config field {field} must be of type {expected_type.__name__}")
    
    if cfg.depth_measurement not in ['relative', 'metric']:
        raise ValueError("depth_measurement must be 'relative' or 'metric'")
    
    if cfg.encoder != 'vitl':
        raise ValueError("Currently only 'vitl' encoder is supported")
    
    if cfg.dataset not in ['hypersim', 'vkitti']:
        raise ValueError("dataset must be 'hypersim' or 'vkitti'")
    
    if cfg.max_depth <= 0:
        raise ValueError("max_depth must be positive")
    
    return Config(
        input_folder=cfg.input_folder,
        output_dir=cfg.output.dir,
        depth_measurement=cfg.depth_measurement,
        encoder=cfg.encoder,
        dataset=cfg.dataset,
        max_depth=cfg.max_depth
    )

def find_repository_root() -> Path:
    """Find the repository root by looking for specific repository markers."""
    current_dir = Path().resolve()
    
    # Keep going up until we find the repository root
    while current_dir != current_dir.parent:
        # Check for common repository markers
        if any((current_dir / marker).exists() for marker in ['.git', 'setup.py', 'pyproject.toml']):
            return current_dir
        current_dir = current_dir.parent
    
    raise RuntimeError("Could not find repository root. Make sure you're running from within the repository.")

def setup_paths(depth_measurement: str) -> Tuple[Path, Path]:
    """Setup paths based on depth measurement type."""
    repo_root = find_repository_root()
    base_path = repo_root / 'models/Depth-Anything-V2'
    
    if depth_measurement == 'metric':
        depth_anything_path = base_path / 'metric_depth'
    else:
        depth_anything_path = base_path
    
    #sys.path.append(str(base_path))
    if depth_measurement == 'metric':
        sys.path.append(str(depth_anything_path))
        logging.info(f"Using metric depth model from {depth_anything_path}")
    elif depth_measurement == 'relative':
        sys.path.append(str(base_path))
        logging.info(f"Using relative depth model from {base_path}")
    
    return repo_root, depth_anything_path

def validate_paths(repo_root: Path, config: Config) -> Tuple[Path, Path, Path]:
    # Convert all paths to be relative to repository root
    input_path = (repo_root / config.input_folder).resolve()
    output_path = (repo_root / config.output_dir).resolve()
    csv_path = (repo_root / config.output_dir / "depthmap_statistics.csv").resolve()
    
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    return input_path, output_path, csv_path

def load_model(depth_anything_path: Path, 
               depth_measurement: str,
               encoder: str = 'vitl',
               dataset: str = 'hypersim',
               max_depth: int = 20) -> Tuple[Any, str]:
    """Unified model loading function for both relative and metric depth."""
    from depth_anything_v2.dpt import DepthAnythingV2
    
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    
    if encoder not in model_configs:
        raise ValueError(f"Unsupported encoder: {encoder}")
    
    # Configure model based on measurement type
    model_config = model_configs[encoder].copy()
    if depth_measurement == 'metric':
        model_config['max_depth'] = max_depth
    
    # Initialize model
    model = DepthAnythingV2(**model_config)
    
    # Determine checkpoint path based on measurement type
    if depth_measurement == 'metric':
        checkpoint_name = f'depth_anything_v2_metric_{dataset}_{encoder}.pth'
    else:
        checkpoint_name = f'depth_anything_v2_{encoder}.pth'
    
    checkpoint = depth_anything_path / 'checkpoints' / checkpoint_name
    
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint}")
    
    logging.info(f"Loading {depth_measurement} depth model with encoder {encoder}")
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    return model, device

def infer_depth(model: Any, image_path: Path) -> Optional[np.ndarray]:
    logging.info(f"Estimating depth for {image_path}")
    raw_img = cv2.imread(str(image_path))
    
    if raw_img is None:
        logging.error(f"Failed to read image: {image_path}")
        return None
    
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        depth = model.infer_image(rgb_img)
    return depth

def save_depth_map(output_folder: Path, image_path: Path, depth_map: np.ndarray) -> Path:
    stem = image_path.stem
    depth_image_path = output_folder / "images" / f"{stem}_depth.jpg"
    depth_array_path = output_folder / "arrays" / f"{stem}_depth.npy"
    
    depth_image_path.parent.mkdir(parents=True, exist_ok=True)
    depth_array_path.parent.mkdir(parents=True, exist_ok=True)
    
    normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(depth_image_path), colored)
    np.save(depth_array_path, depth_map)
    
    return depth_image_path

def compute_statistics(depth_map: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(np.percentile(depth_map, 5)),
        float(np.percentile(depth_map, 95)),
        float(np.mean(depth_map)),
        float(np.median(depth_map))
    )

def write_csv(csv_path: Path, data: List[Any], headers: Optional[List[str]] = None) -> None:
    file_exists = csv_path.exists()
    with csv_path.open('a', newline='') as f:
        writer = csv.writer(f)
        if headers and not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

def get_valid_images(folder: Path) -> List[Path]:
    valid_extensions: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    return [p for p in folder.rglob('*') if p.suffix.lower() in valid_extensions]

def process_images(model: Any, input_folder: Path, output_folder: Path, csv_path: Path) -> None:
    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(f"Invalid input folder: {input_folder}")
    
    output_folder.mkdir(parents=True, exist_ok=True)
    headers = ['filename', 'Quantile05', 'Quantile95', 'Mean', 'Median']
    
    for img_path in get_valid_images(input_folder):
        logging.info(f"Processing {img_path.name}")
        depth = infer_depth(model, img_path)
        
        if depth is None:
            continue
            
        stats = compute_statistics(depth)
        write_csv(csv_path, [img_path.name, *stats], headers=headers)
        depth_image_path = save_depth_map(output_folder, img_path, depth)
        logging.info(f"Saved depth map at {depth_image_path}")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    try:
        # Validate configuration
        config = validate_config(cfg)
        
        # Setup paths relative to repository root
        repo_root, depth_anything_path = setup_paths(config.depth_measurement)
        input_path, output_path, csv_path = validate_paths(repo_root, config)
        
        # Load model with unified function
        model, _ = load_model(
            depth_anything_path=depth_anything_path,
            depth_measurement=config.depth_measurement,
            encoder=config.encoder,
            dataset=config.dataset,
            max_depth=config.max_depth
        )
        
        # Process images
        process_images(model, input_path, output_path, csv_path)
        
    except (ValueError, TypeError, FileNotFoundError) as e:
        logging.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
