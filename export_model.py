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
from torchvision import T
from PIL import Image

from preprocessing.preprocess import extract_aligned_face_dlib

import dlib
import numpy as np
import cv2



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

def face_cropper(images, size=256):
    """
    Crop the face from the image based on the bounding box.
    :param image: PIL Image
    :param size: Size to resize the cropped face to
    :return: Cropped and resized face as a PIL Image
    """

    def get_keypts(image, face, predictor, face_detector):
        # detect the facial landmarks for the selected face
        shape = predictor(image, face)
        
        # select the key points for the eyes, nose, and mouth
        leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
        reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
        nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
        lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
        rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
        
        pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

        return pts

    # input RGB image
    def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
        def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
            """ 
            align and crop the face according to the given bbox and landmarks
            landmark: 5 key points
            """

            M = None
            target_size = [112, 112]
            dst = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)

            if target_size[1] == 112:
                dst[:, 0] += 8.0

            dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
            dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

            target_size = outsize

            margin_rate = scale - 1
            x_margin = target_size[0] * margin_rate / 2.
            y_margin = target_size[1] * margin_rate / 2.

            # move
            dst[:, 0] += x_margin
            dst[:, 1] += y_margin

            # resize
            dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
            dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

            src = landmark.astype(np.float32)

            # use skimage tranformation
            tform = trans.SimilarityTransform()
            tform.estimate(src, dst)
            M = tform.params[0:2, :]

            # M: use opencv
            # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

            img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

            if outsize is not None:
                img = cv2.resize(img, (outsize[1], outsize[0]))
            
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
                mask = cv2.resize(mask, (outsize[1], outsize[0]))
                return img, mask
            else:
                return img, None

        # Image size
        height, width = image.shape[:2]

        # Convert to rgb
        # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect with dlib
        faces = face_detector(rgb, 1)
        if len(faces):
            # For now only take the biggest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get the landmarks/parts for the face in box d only with the five key points
            landmarks = get_keypts(rgb, face, predictor, face_detector)

            # Align and crop the face
            cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
            # cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
            
            # Extract the all landmarks from the aligned face
            # face_align = face_detector(cropped_face, 1)
            # if len(face_align) == 0:
            #     return None

            return cropped_face
        
        else:
            return None

    # Define face detector and predictor models
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = str(Path(__file__).parent / 'preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat')
    ## Check if predictor path exists
    if not os.path.exists(predictor_path):
        logger.error(f"Predictor path does not exist: {predictor_path}")
        sys.exit()
    predictor = dlib.shape_predictor(predictor_path)

    # stores processed faces in rbg format
    processed_faces = []

    # Check if the input is a list of  PIL Image or a numpy array
    if isinstance(images, Image.Image):
        images = [images]  # Convert single image to list
        
    if isinstance(images, np.ndarray):
        # assume loaded using OpenCV
        images = [images]  # Convert single image to list
    
    # for PIL Images
    if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
        for image in images:
            processed_face, _, _ = _process_single_image(image, face_detector, predictor, size)
            # Convert RGB numpy array to PIL Image
            if processed_face is not None:
                processed_face = Image.fromarray(processed_face)
            processed_faces.append(processed_face)

    # for openCV images
    elif isinstance(images, list) and all((isinstance(img, np.ndarray) and img.ndim == 3 and imgeshape[2] == 3) for img in images):
        for image in images:
            # OpenCV loads images in BGR format, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_face, _, _ = _process_single_image(image, face_detector, predictor, size)
            if processed_face is not None:
                processed_face = Image.fromarray(processed_face)
            processed_faces.append(processed_face)
    else:
        raise ValueError("Input images must be a list of PIL Images or numpy arrays with 3 channels.")

    # returned a list PIL Images of processed faces
    return processed_faces


    
