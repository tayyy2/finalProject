import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for model input.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Desired output image size
    
    Returns:
        numpy.ndarray: Preprocessed image tensor ready for model prediction
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image to target size
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    normalized_image = resized_image / 255.0
    
    # Reshape for model input (add batch and channel dimensions)
    processed_image = normalized_image.reshape(1, target_size[0], target_size[1], 1)
    
    return processed_image