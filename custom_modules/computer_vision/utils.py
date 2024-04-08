import numpy as np
import base64
import cv2

def B64_to_cv2(image_string):
    """
    Converts a base64-encoded image string to a cv2 image object.

    :param image_string: The base64-encoded image string.
    :return: The decoded image as a cv2 image object.
    """
    image_bytes = base64.b64decode(image_string)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_object = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
    return image_object

def cv2_to_B64(image_object):
    """
    Converts a cv2 image object to a base64-encoded image string.

    :param image_object: The cv2 image object to encode.
    :return: The image as a base64-encoded string.
    """
    _, image_array = cv2.imencode('.jpg', image_object)
    image_bytes = image_array.tobytes()
    image_string = base64.b64encode(image_bytes)
    return image_string

def load_image(image_path):
    """
    Loads an image from the specified file path.

    :param image_path: The path to the image file.
    :return: The image as a cv2 image object.
    """
    image = cv2.imread(image_path)
    return image

def save_image(image, image_path):
    """
    Saves an image to the specified file path.

    :param image: The image to save.
    :param image_path: The path to save the image file.
    """
    cv2.imwrite(image_path, image)
