import os
import shutil
import sys
from preprocessing.preprocess import main as preprocess
from preprocessing.rearrange import main as rearrange
from training.test import main as tests
from pathlib import Path
import argparse
from detector_map import DETECTOR_MAP

# Use all models if '-m all' is specified


if __name__ == '__main__':
    
    # you also can choose not to use arguments and set the model and data path directly in the code
    
    parser = argparse.ArgumentParser(description="Process a test dataset using a selected model.")
    parser.add_argument('-m', '--model', type=str, nargs='+', default=['xception'], help='Name(s) of the model(s) to use (space separated)')
    parser.add_argument('-d', '--dataset-path', type=str, default='../datasets/TestSet', help='Path to the dataset directory')
    parser.add_argument('-p', '--preprocess', action='store_true', help='Enable preprocessing (set flag to activate)')

    args = parser.parse_args()

    models = args.model
    data_path = Path(args.dataset_path).resolve()
    PREPROCESS = args.preprocess
    
    print("target path:", data_path)

    if "all" in models:
        # If 'all' is specified, use all available models
        models = list(DETECTOR_MAP.keys())
        print("Using all available models:")
        print(', '.join(models))
        print("Other model names are ignored.")
    else:
        # Check if every model is valid
        invalid_models = [m for m in models if m not in DETECTOR_MAP]
        if invalid_models:
            print("Invalid model(s):", ', '.join(invalid_models))
            print("Available models:")
            for model_name in DETECTOR_MAP.keys():
                print(f"- {model_name}")
            sys.exit(1)

    for idx, model in enumerate(models):
        detector_config = Path(DETECTOR_MAP[model]['config']).resolve()
        detector_weights = Path(DETECTOR_MAP[model]['weights']).resolve()

        if idx == 0:
            if PREPROCESS:                
                print("Preprocessing Stage 1: Generating Frames and Landmarks!")
                sys.stdout.flush()
                preprocess(data_path)
                print("Frames and Landmarks Generated!")
                sys.stdout.flush()
                print("Preprocessing Stage 2: Generating Json file for testing!")
                rearrange(data_path)
                print("JSON File Generated!")
                sys.stdout.flush()
            else:
                print("Skipping Preprocessing Stage! As Preprocess is set to False.")
                sys.stdout.flush()

        print(f"\n==== Testing Started for model {model} ====\n")
        tests(detector_path=detector_config, test_datasets=["TestSet"], weights_path=detector_weights)
        print(f"Results Generated in {Path(__file__).parent / 'results' / model} for model {model}")
        sys.stdout.flush()
        print(f"\n==== Testing Ended for model {model} ====\n")
        sys.stdout.flush()
