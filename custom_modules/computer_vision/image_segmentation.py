import numpy as np
import cv2
import os

class ImageMask:
    def __init__(self, mask, size=(1920, 1080)):
        """
        Initializes the ImageMask class.

        :param mask: A binary mask array.
        :param size: The target size of the mask.
        """
        if mask.ndim == 2:
            mask = np.stack((mask, mask, mask), axis=-1)
        self.mask = mask
        self.size = size

    def apply(self, image):
        """
        Applies the mask to the given image.

        :param image: Image to which the mask will be applied.
        :return: The masked image.
        """
        result = np.zeros_like(image)
        result[self.mask] = image[self.mask]
        return result

    def save(self, mask_path):
        """
        Saves the mask as an image file.

        :param mask_path: Path to save the mask image.
        """
        image = np.zeros(self.size[::-1])
        image[self.mask[:, :, 0]] = 255
        cv2.imwrite(mask_path, image)

    @staticmethod
    def __draw_on_event(mask_data, event, x, y, *args):
        if event == cv2.EVENT_LBUTTONDOWN:
            mask_data['start'] = (x, y)
            mask_data['is_drawing'] = True
        elif event == cv2.EVENT_MOUSEMOVE and mask_data['is_drawing']:
            cv2.line(mask_data['image'], mask_data['start'], (x, y), mask_data['color'], 2)
            mask_data['start'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mask_data['is_drawing'] = False
            cv2.line(mask_data['image'], mask_data['start'], (x, y), mask_data['color'], 2)

    @staticmethod
    def from_file(mask_path, size=(1920, 1080)):
        """
        Creates an ImageMask from a file.

        :param mask_path: Path to the mask image file.
        :param size: Size to which the mask will be resized.
        :return: An ImageMask object.
        """
        if not os.path.isfile(mask_path):
            raise FileNotFoundError("Mask file not found.")
        mask_image = cv2.imread(mask_path)
        mask_image = cv2.resize(mask_image, size)
        mask = mask_image > 0
        return ImageMask(mask=mask, size=size)

    @staticmethod
    def create_new(size=(1920, 1080), ref_img=None, clr=(0, 0, 255)):
        """
        Interactively creates a new mask.

        :param size: Size of the mask to be created.
        :param ref_img: Reference image for mask creation, if any.
        :param clr: Color for the drawing tool.
        :return: An ImageMask object.
        """
        mask_data = {
            'image': np.zeros(size[::-1]) if ref_img is None else np.copy(ref_img),
            'color': clr,
            'start': None,
            'is_drawing': False,
        }
        mouse_callback = lambda *args: \
            ImageMask.__draw_on_event(mask_data, *args)
        cv2.namedWindow('Mask Creator')
        cv2.setMouseCallback('Mask Creator', mouse_callback)
        active = True
        while active:
            cv2.imshow('Mask Creator', mask_data['image'])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                active = False
        cv2.destroyAllWindows()
        fig_mask = cv2.inRange(mask_data['image'], clr, clr)
        contours, _ = cv2.findContours(fig_mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(size[::-1])
        cv2.drawContours(image, contours, -1, (255, 255, 255), 
            thickness=cv2.FILLED)
        cv2.imwrite('mask.png', image)
        mask = image > 0
        return ImageMask(mask=mask, size=size)


class ImageSegmentator:
    def __init__(self, size=(1920, 1080), segments=[]):
        """
        Initializes the ImageSegmentator class.

        :param size: Size of the images that will be segmented.
        :param segments: List of paths to mask images for segmentation.
        """
        self.size = size
        self.image_masks = [ImageMask.from_file(mask_path, size) for mask_path in segments]
        self.segment_count = len(self.image_masks)

    def segment_image(self, frame):
        """
        Segments an image according to the predefined masks.

        :param frame: Image to be segmented.
        :yield: Segmented parts of the image.
        """
        if self.segment_count == 0:
            yield frame
            return
        for mask in self.image_masks:
            yield mask.apply(frame)
