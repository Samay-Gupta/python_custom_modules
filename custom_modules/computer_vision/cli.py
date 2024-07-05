from custom_modules.utils import get_default_config

from custom_modules.computer_vision.utils import load_image, save_image

import argparse
import json
import sys
import os

INVALID_MODE_ERROR = "[ERROR] Invalid Mode"

INVALID_USAGE_ERROR = "[ERROR] Invalid Usage"

TOOL_USAGE_TEXT = """
[USAGE] cv <mode> <args>

mode: The operation mode to be performed.
    - 'classify': Classify objects in an image.
    - 'read': Extract text from an image.
    - 'scale': Scale an image by a factor of 2, 3, or 4.
    - 'scan': Read QR and barcodes from an image.
        
"""

SCAN_USAGE_TEXT = """
[USAGE] cv scan <image_path> <code_type>

image_path: The path to the image file from which QR and barcodes need to be extracted.

code_type: The type of code to be scanned.
    - 'qr': Scan QR codes. [default]
    - 'barcode': Scan barcodes.
"""

SCALE_USAGE_TEXT = """
[USAGE] cv read <image_path> <scale> <output>

image_path: The path to the image file from which text needs to be extracted.

scale: scale factor to resize the image.
    - Allowed values: 2, 3, 4
    - Default value: 2

output: The path to the output image file.
    - Default value: output.png
"""

OCR_USAGE_TEXT = """
[USAGE] cv read <image_path> <preprocess>

image_path: The path to the image file from which text needs to be extracted.

preprocess: The preprocessing mode to be applied to the image before OCR.
    - 'blur': Applies median blur to the image. [default]
    - 'threshold': Applies binary thresholding with Otsu's method.
"""

CLASSIFY_USAGE_TEXT = """
[USAGE] cv classify <image_path> <output>

image_path: The path to the image file from which text needs to be extracted.

output: The path to the output image file.
"""

class ComputerVisionCLI:
    def __init__(self, config) -> None:
        """
        To initialize an instance of ComputerVisionApp
        """
        self.config = config
        self.model_dir = os.path.join(
            config["storageDir"] or os.getcwd(),
            config["modelDir"]
        )

    def run(self):
        args = self.get_args()
        if args["mode"] == "classify":
            from custom_modules.computer_vision.object_detection import ObjectDetector
            model_dir = os.path.join(
                self.model_dir, 
                self.config["models"]["objectDetection"]["modelDir"]
            )
            clf = ObjectDetector(model_dir)
            if args["live"]:
                clf.classify_from_camera()
            else:
                image = load_image(args["source"])
                output = clf.classify_and_draw(image)
                save_image(output, args["output"])
        if args["mode"] == "read":
            from custom_modules.computer_vision.ocr import TextRecognizer
            image = load_image(args["source"])
            text = TextRecognizer.from_image(image, args["preprocess"])
            print(text)
        if args["mode"] == "scale":
            from custom_modules.computer_vision.image_scaler import ImageScaler
            model_dir = os.path.join(
                self.model_dir, 
                self.config["models"]["imageScaler"]["modelDir"]
            )
            image_scaler = ImageScaler(model_dir, args["scale"])
            image = load_image(args["source"])
            scaled_image = image_scaler.scale(image)
            save_image(scaled_image, args["output"])
        if args["mode"] == "scan":
            from custom_modules.computer_vision.code_scanners import CodeScanner
            image = load_image(args["source"])
            text = CodeScanner.scan(image, args["code_type"])
            print(text)

    def get_args(self):
        if len(sys.argv) < 2:
            print(INVALID_MODE_ERROR)
            print(TOOL_USAGE_TEXT)
            sys.exit(1)
        filename, mode, *arg_list = sys.argv
        argc = len(arg_list)
        args = { "mode": mode }
        if mode == "classify":
            if argc > 2:
                print(INVALID_USAGE_ERROR)
                print(CLASSIFY_USAGE_TEXT)
                sys.exit(1)
            args["live"] = argc == 0
            if argc > 0:
                args["source"] = arg_list[0]
                args["output"] = "output.png"
                args["live"] = False
            if argc > 1:
                args["output"] = arg_list[1]
            return args
        if mode == "read":
            if not (1 <= argc <= 2):
                print(INVALID_USAGE_ERROR)
                print(OCR_USAGE_TEXT)
                sys.exit(1)
            args["source"] = arg_list[0]
            args["preprocess"] = "blur"
            if argc == 2:
                args["preprocess"] = arg_list[1]
            return args
        if mode == "scale":
            if not (1 <= argc <= 3):
                print(INVALID_USAGE_ERROR)
                print(SCALE_USAGE_TEXT)
                sys.exit(1)
            args["source"] = arg_list[0]
            args["scale"] = 2
            args["output"] = "output.png"
            if argc == 2 and arg_list[1] in ["2", "3", "4"]:
                args["scale"] = int(arg_list[1])
            return args
        if mode == "scan":
            if not (1 <= argc <= 2):
                print(INVALID_USAGE_ERROR)
                print(SCAN_USAGE_TEXT)
                sys.exit(1)
            args["source"] = arg_list[0]
            args["code_type"] = "qr"
            if argc == 2 and arg_list[1] in ["qr", "barcode"]:
                args["code_type"] = arg_list[1]
            return args
        print(INVALID_USAGE_ERROR)
        print(TOOL_USAGE_TEXT)
        sys.exit(1)
    
    @staticmethod
    def from_config_file(config_path: str = None):
        if config_path is None:
            config = get_default_config()
        else:
            with open(config_path, 'r') as fp:
                config = json.load(fp)
        return ComputerVisionCLI(config)

if __name__ == '__main__':
    config = get_default_config()
    app = ComputerVisionCLI(config)
    app.run()
