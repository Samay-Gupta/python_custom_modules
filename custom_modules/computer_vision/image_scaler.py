from cv2 import dnn_superres
import cv2
import os

class ImageScaler:
    def __init__(self, model_dir=None, scale=2) -> None:
        """
        Initializes the ImageScaler with a specified super-resolution model.

        :param model_dir: Directory where the super-resolution model is stored.
                          If None, uses the current working directory.
        :param scale: The scale factor for image upscaling. Valid options are 2, 3, or 4.
                      Default is 2.
        :raises ValueError: If an invalid scale factor is provided.
        """
        if scale not in (2, 3, 4):
            raise ValueError(f"Invalid Predefined Scale - {scale}!")

        if model_dir is None:
            model_dir = os.getcwd()
        model_path = os.path.join(model_dir, f"EDSR_x{scale}.pb")
        self.model = dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(model_path)
        self.model.setModel("edsr", scale)

    def scale(self, image):
        """
        Scales an image using the super-resolution model.

        :param image: The input image to be upscaled.
        :return: The upscaled image.
        """
        return self.model.upsample(image)
