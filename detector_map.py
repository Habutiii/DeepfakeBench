from pathlib import Path

# Update paths to be relative to the file location
BASE_DIR = Path(__file__).resolve().parent

DETECTOR_MAP = {
    'xception': {
        "config": BASE_DIR / "training/config/detector/xception.yaml",
        "weights": BASE_DIR / "training/pretrained/xception_best.pth"
    },
    'resnet': {
        "config": BASE_DIR / "training/config/detector/resnet34.yaml",
        "weights": BASE_DIR / "training/pretrained/cnnaug_best.pth"
    },
    'efficientnetb4': {
        "config": BASE_DIR / "training/config/detector/efficientnetb4.yaml",
        "weights": BASE_DIR / "training/pretrained/effnb4_best.pth"
    },
    'meso4': {
        "config": BASE_DIR / "training/config/detector/meso4.yaml",
        "weights": BASE_DIR / "training/pretrained/meso4_best.pth"
    },
    'meso4Inception': {
        "config": BASE_DIR / "training/config/detector/meso4Inception.yaml",
        "weights": BASE_DIR / "training/pretrained/meso4Incep_best.pth"
    },
    # 'core': {
    #     "config": BASE_DIR / "training/config/detector/core.yaml",
    #     "weights": BASE_DIR / "training/pretrained/core_best.pth"
    # },
    'ucf': {
        "config": BASE_DIR / "training/config/detector/ucf.yaml",
        "weights": BASE_DIR / "training/pretrained/ucf_best.pth"
    },
    'ffd': {
        "config": BASE_DIR / "training/config/detector/ffd.yaml",
        "weights": BASE_DIR / "training/pretrained/ffd_best.pth"
    },
    'capsule': {
        "config": BASE_DIR / "training/config/detector/capsule_net.yaml",
        "weights": BASE_DIR / "training/pretrained/capsule_best.pth"
    },
    'recce': {
        "config": BASE_DIR / "training/config/detector/recce.yaml",
        "weights": BASE_DIR / "training/pretrained/recce_best.pth"
    },
    'f3net': {
        "config": BASE_DIR / "training/config/detector/f3net.yaml",
        "weights": BASE_DIR / "training/pretrained/f3net_best.pth"
    },
    'spsl': {
        "config": BASE_DIR / "training/config/detector/spsl.yaml",
        "weights": BASE_DIR / "training/pretrained/spsl_best.pth"
    },
    'srm': {
        "config": BASE_DIR / "training/config/detector/srm.yaml",
        "weights": BASE_DIR / "training/pretrained/srm_best.pth"
    # },
    # 'altfreezing': {
    #     "config": BASE_DIR / "training/config/detector/altfreezing.yaml",
    #     "weights": BASE_DIR / "training/pretrained/I3D_8x8_R50.pth"
    }
}
