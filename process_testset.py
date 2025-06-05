import os
import shutil
import sys
from preprocessing.preprocess import main as preprocess
from preprocessing.rearrange import main as rearrange
from training.test import main as tests
from pathlib import Path

detector_map = {
    
    ## Naive Models
    'xception': {
        "config":"./training/config/detector/xception.yaml",
        "weights":"./training/pretrained/xception_best.pth"
    },
    
    'resnet': {
        "config":"./training/config/detector/resnet34.yaml",
        "weights":"./training/pretrained/cnnaug_best.pth"
    },
    
    'efficientnetb4': {
        "config": "./training/config/detector/efficientnetb4.yaml",
        "weights": "./training/pretrained/effnb4_best.pth"
    },
    
    'meso4': {
        "config": "./training/config/detector/meso4.yaml",
        "weights": "./training/pretrained/meso4_best.pth"
    },
    
    'meso4Inception': {
        "config": "./training/config/detector/meso4Inception.yaml",
        "weights": "./training/pretrained/meso4Incep_best.pth"
    },
    
    # Spatial Models
    
    
    'core': {  # Xception based model
        "config":"./training/config/detector/core.yaml",
        "weights":"./training/pretrained/core_best.pth"
    },
    'ucf': {  #  Xception based model
        "config": "./training/config/detector/ucf.yaml",
        "weights": "./training/pretrained/ucf_best.pth"
    },
    
    'ffd' : {  # Xception based model
        "config": "./training/config/detector/ffd.yaml",
        "weights": "./training/pretrained/ffd_best.pth"
    },
    
    'capsule': {   # Capsule Network based model
        "config": "./training/config/detector/capsule_net.yaml",
        "weights": "./training/pretrained/capsule_best.pth"
    },
    
    'recce': {  # Designed based model
        "config": "./training/config/detector/recce.yaml",
        "weights": "./training/pretrained/recce_best.pth"
    },
    
    
    # Frequency Models
    # All based on Xception
    'f3net': {
        "config": "./training/config/detector/f3net.yaml",
        "weights": "./training/pretrained/f3net_best.pth"
    },
    'spsl' : {
        "config": "./training/config/detector/spsl.yaml",
        "weights": "./training/pretrained/spsl_best.pth"
    },
    'srm': {
        "config": "./training/config/detector/srm.yaml",
        "weights": "./training/pretrained/srm_best.pth"
    },
    
    # Video Models
    'altfreezing': {
        "config": "./training/config/detector/altfreezing.yaml",
        "weights": "./training/pretrained/I3D_8x8_R50.pth"
    },
    
    
}


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python process_testset.py <model_name>")
        print("Available models:")
        for model in detector_map.keys():
            print(f"- {model}")
        sys.exit(1)
        
    model = sys.argv[1]
    
    PREPROCESS = False
    
    datasets = ["TestSet"]

    detector_config = Path(detector_map[model]['config']).resolve()
    detector_weights = Path(detector_map[model]['weights']).resolve()

    if PREPROCESS:
        preprocess()
        print("Stage 1: Frames and Landmarks Generated!")
        sys.stdout.flush()
        
        rearrange()
        print("Stage 2: JSON File Generated!")
        sys.stdout.flush()
        
    else:
        print("Skipping Preprocessing Stage! As Preprocess is set to False.")
        sys.stdout.flush()
        
    
    print("Stage 3: Testing Started!")
    tests(detector_path=detector_config, test_datasets=datasets, weights_path=detector_weights)
    # shutil.move(os.path.join(os.getcwd(), f'./results/{model}/TestSet_results.csv'), os.path.join(src_dir, 'frame.csv'))
    # shutil.move(os.path.join(os.getcwd(), f'./results/{model}/TestSet_video_results.csv'), os.path.join(src_dir, 'video.csv'))
    print("Stage 4: Results Generated!")
    sys.stdout.flush()
