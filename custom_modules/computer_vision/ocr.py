import pytesseract
import cv2

class TextRecognizer:
    @staticmethod
    def from_image(image, mode: str = "blur"):
        """
        Extracts text from an image using OCR (Optical Character Recognition).

        The function preprocesses the image based on the specified mode before applying OCR.
        The preprocessing step is crucial for enhancing the OCR accuracy.

        :param image: The input image from which text needs to be extracted.
        :param mode: The preprocessing mode. Can be 'blur' or 'threshold'.
                     'blur' applies median blur to the image.
                     'threshold' applies binary thresholding with Otsu's method.
                     Default is 'blur'.
        :return: A list of strings, where each string represents a line of text detected in the image.
        """
        preprocessed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if mode == "blur":
            preprocessed = cv2.medianBlur(preprocessed, 3)
        elif mode == "threshold":
            preprocessed = cv2.threshold(preprocessed, 0, 255,
                                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        ocr_output = pytesseract.image_to_string(preprocessed)
        return ocr_output
