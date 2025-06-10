# Ensure the detectors module is imported correctly
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "training"))

import torch
import yaml
from detectors import DETECTOR
# Import DETECTOR_MAP from detector_map.py
from detector_map import DETECTOR_MAP



# Update paths to be relative to the file location
BASE_DIR = Path(__file__).resolve().parent

def load_model(detector_name):
    if detector_name not in DETECTOR_MAP:
        raise ValueError(f"Invalid detector name. Available detectors: {list(DETECTOR_MAP.keys())}")

    detector_config_path = Path(DETECTOR_MAP[detector_name]['config']).resolve()
    detector_weights_path = Path(DETECTOR_MAP[detector_name]['weights']).resolve()

    # Load configuration
    with open(detector_config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['pretrained'] = str(detector_weights_path)
    config['weights'] = str(detector_weights_path)
    # Prepare model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load weights
    try:
        ckpt = torch.load(detector_weights_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(ckpt, strict=True)
        print(f"Model '{detector_name}' loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load weights for model '{detector_name}': {e}")

    return model

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a loaded model.")
    parser.add_argument('-m', '--model', type=str, required=True, help='Name of the model to load')

    args = parser.parse_args()

    model = load_model(args.model)
    print(f"Model '{args.model}' is ready for inference.")
