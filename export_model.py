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
from torchvision import transforms as T
from PIL import Image
import dlib
import numpy as np
import cv2
import os

from skimage import transform as trans

from preprocessing.preprocess import extract_aligned_face_dlib

from tqdm import tqdm
import matplotlib.pyplot as plt




# Update paths to be relative to the file location
BASE_DIR = Path(__file__).resolve().parent


# Example usage
# choose a model name from DETECTOR_MAP

# Some config settings you may consider passing
# extra_config = {
#   'test_batchSize': 32,
#   'workers': 4,
#   'backbone_config': {
#        dropout: true
#   }
#}

# model, process, config = load_model(args.model)
# model.eval()
# image_paths = [...]
# images = [Image.open(path) for path in image_paths]  # Load images as PIL Images
# processed_images = process(images)
# model_output = model(processed_images)
# logit = model_output['cls']
# scores = model_output['prob']
# labels = model_output['label']

def load_model(detector_name, extra_config={}):
    if detector_name not in DETECTOR_MAP:
        raise ValueError(f"Invalid detector name. Available detectors: {list(DETECTOR_MAP.keys())}")

    detector_config_path = Path(DETECTOR_MAP[detector_name]['config']).resolve()
    detector_weights_path = Path(DETECTOR_MAP[detector_name]['weights']).resolve()

    # Load configuration
    with open(detector_config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['pretrained'] = str(detector_weights_path)
    config['weights'] = str(detector_weights_path)


    #merge extra config
    config.update(extra_config)


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


    # Process input images
    # Accepts both PIL Images and PyTorch tensors
    # force_resize: if True, will force resize images to the model's input size, but may cause loss of quality
    def process(images, force_resize=False):
        """
        Process images for the model.
        This function should be adapted based on the model's requirements.
        """
        # Example processing: resize, normalize, based on model config
        size = config.get('resolution')
        mean = config.get('mean')
        std = config.get('std')

        # if is a single image
        if isinstance(images, Image.Image):
            images = [images]

        # if images are in PIL Image format
        if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            if not force_resize and any(img.size != (size, size) for img in images):
                raise ValueError(f"Input images must be of size {size}x{size}, or turn on force resize.")
            transform = T.Compose([
                T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])

            return torch.stack([transform(img) for img in images])

        # if images are in tensor format
        elif isinstance(images, torch.Tensor):
            # check if images are in (B, C, H, W) format
            if images.dim() == 3 and images.size(0) == 3:
                # Single image tensor is in (C, H, W) format
                images = images.unsqueeze(0)
            
            # check if images are in (B, C, H, W) format
            if images.dim() != 4 or images.size(1) != 3:
                raise ValueError("Input images must be in (B, C, H, W) format with 3 channels.")
            
            # check if images are of the correct size
            if images.size(2) != size or images.size(3) != size:
                # resize it
                if(force_resize):
                    # print(f"Resizing images to {size}x{size} as per model requirements.")
                    images = T.functional.interpolate(images, size=(size, size), mode='bicubic', align_corners=False)
                else:
                    raise ValueError(f"Input images must be of size {size}x{size}, or turn on force resize.")

            # check if image is [0,1] range or [0,255] range
            if images.min() < 0.0:
                print(images.min())
                raise ValueError("Input images must be in [0, 1] or [0, 255] range.")

            # if images are in [0, 255] range, convert to [0, 1] range
            # [0,1] is never uint8 format
            if images.dtype == torch.uint8 or torch.any(images > 1.0):
                # convert to [0, 1] range
                images = images / 255.0
                
            # set transformation
            transform = T.Compose([
                T.Normalize(mean=mean, std=std)
            ])

            return transform(images)


        else:
            raise ValueError("Input images must either be a PIL Image list or a PyTorch tensor in (B, C, H, W) format.")

    return model, process, config




# the input image can be a PIL Image or a numpy array

# Optimize face cropping by using multiprocessing
from multiprocessing import Pool, cpu_count

# Move process_image to the global scope
def process_image(image, size):
    """
    Process a single image: detect, crop, and resize the face.
    :param image: Input image as a numpy array (BGR format).
    :param size: Size to resize the cropped face to.
    :return: Resized face and bounding box.
    """
    # Initialize the face detector
    face_detector = dlib.get_frontal_face_detector()

    def extract_face_bbox(face_detector, image):
        """
        Detect the face and calculate the bounding box.
        :param face_detector: Dlib face detector.
        :param image: Input image as a numpy array (BGR format).
        :return: Cropped face and bounding box coordinates (x_min, y_min, x_max, y_max).
        """
        # Detect faces in the image
        faces = face_detector(image, 1)

        if len(faces) > 0:
            # Select the largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())

            # Calculate bounding box coordinates
            x_min = max(0, face.left())
            y_min = max(0, face.top())
            x_max = min(image.shape[1], face.right())
            y_max = min(image.shape[0], face.bottom())

            # Crop the face
            cropped_face = image[y_min:y_max, x_min:x_max]

            return cropped_face, (x_min, y_min, x_max, y_max)

        return None, None

    # Detect and crop the face
    cropped_face, bbox = extract_face_bbox(face_detector, image)

    if cropped_face is not None:
        # Debugging: Check the range of cropped face values
        
        if cropped_face is not None and cropped_face.min() < 0:
            print("[DEBUG] Checking cropped face values in process_image:")
            print("Cropped face min:", cropped_face.min())

        # Resize the cropped face
        resized_face = cv2.resize(cropped_face, (size, size))
        # Convert to RGB format if needed
        resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        return resized_face, bbox
    else:
        return None, None

# Update process_image_with_size to include indices
def process_image_with_index(args):
    index, image, size = args
    cropped_face, bbox = process_image(image, size)
    return index, cropped_face, bbox

# Define a helper function for processing images with size
def process_image_with_size(args):
    image, size = args
    return process_image(image, size)

# Update face_cropper to sort results by index
def face_cropper(images, size=256):
    """
    Crop the face from the image based on the bounding box.
    :param images: List of images as numpy arrays (BGR format).
    :param size: Size to resize the cropped face to.
    :return: Cropped and resized (RGB) face as a numpy array and the bounding box coordinates.
    """
    results = []

    with Pool(cpu_count()) as pool:
        with tqdm(total=len(images), desc="Cropping Faces") as pbar:
            for result in pool.imap_unordered(process_image_with_index, [(i, image, size) for i, image in enumerate(images)]):
                results.append(result)
                pbar.update(1)

    # Sort results by index to maintain input order
    results.sort(key=lambda x: x[0])
    cropped_faces, bounding_boxes = zip(*[(res[1], res[2]) for res in results])

    return list(cropped_faces), list(bounding_boxes)


from concurrent.futures import ThreadPoolExecutor

def face_paster(processed_faces, bounding_boxes, original_images):
    """
    Paste the processed faces back to the original images using multi-threading.
    :param processed_faces: List of processed faces as numpy arrays.
    :param bounding_boxes: List of bounding box coordinates (x_min, y_min, x_max, y_max).
    :param original_images: List of original images as numpy arrays (BGR format).
    :return: List of images with pasted faces.
    """
    def paste_face(face, bbox, original):
        if face is not None and bbox is not None:
            try:
                # Ensure face is a valid NumPy array
                if not isinstance(face, np.ndarray):
                    # convert to numpy array if it's a PIL Image
                    if isinstance(face, Image.Image):
                        face = np.array(face)
                        # Convert to BGR format if needed
                        if face.ndim == 3 and face.shape[2] == 3:
                            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    else:
                        raise ValueError("Invalid face data: not a NumPy array or PIL Image, got" + str(type(face)))

                x_min, y_min, x_max, y_max = bbox

                # Ensure bounding box is within image dimensions
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(original.shape[1], x_max)
                y_max = min(original.shape[0], y_max)

                # Resize the processed face to match the bounding box size
                resized_face = cv2.resize(face, (x_max - x_min, y_max - y_min))

                # Paste the resized face onto the original image
                original[y_min:y_max, x_min:x_max] = resized_face

            except Exception as e:
                print(f"Error pasting face: {e}")

        return original

    pasted_images = []
    with ThreadPoolExecutor() as executor:
        with tqdm(total=len(processed_faces), desc="Pasting Faces") as pbar:
            for result in executor.map(lambda args: paste_face(*args), zip(processed_faces, bounding_boxes, original_images)):
                pasted_images.append(result)
                pbar.update(1)

    return pasted_images
