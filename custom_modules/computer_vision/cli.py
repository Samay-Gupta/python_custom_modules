from ..utils import get_default_config, get_base_dir, get_requirements

from .utils import load_image, save_image

import argparse
import json
import sys
import os

INVALID_USAGE_ERROR = "    [ERROR] Invalid Usage"

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
            pass
        if args["mode"] == "read":
            image = load_image(args["source"])
            text = TextRecognizer.from_image(image, args["preprocess"])
            print(text)
        if args["mode"] == "scale":
            model_dir = os.path.join(
                self.model_dir, 
                "image_scaler"
            )
            image_scaler = ImageScaler(model_dir, args["scale"])
            image = load_image(args["source"])
            scaled_image = image_scaler.scale(image)
            save_image(scaled_image, args["output"])
        if args["mode"] == "scan":
            image = load_image(args["source"])
            text = CodeScanner.scan(image, args["code_type"])
            print(text)

    def get_args(self):
        if len(sys.argv) < 2:
            print(sys.argv)
            print("[ERROR] Required mode")
            sys.exit(1)
        filename, mode, *arg_list = sys.argv
        argc = len(arg_list)
        args = { "mode": mode }
        if mode == "classify":
            ap = argparse.ArgumentParser()
            ap.add_argument('-src', '--source', type=str, default=None,
                help="Set input source")
            ap.add_argument('-l', '--live', action='store_true', default=False,
                help="Use video as source")
            ap.add_argument('-m', '--model', default='all', choices=['face', 'object', 'all'],
                help="Classifier list")
            args = vars(ap.parse_args(arg_list))
            args["live"] = args["source"] is None
            return args
        if mode == "read":
            from .ocr import TextRecognizer
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
            from .image_scaler import ImageScaler
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
            from .code_scanners import CodeScanner
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
        print(f"[USAGE] python3 {filename} <mode> <args>")
        sys.exit(1)
    
    @staticmethod
    def from_config_file(config_path: str = None):
        if config_path is None:
            config = get_default_config()
        else:
            with open(config_path, 'r') as fp:
                config = json.load(fp)
        return ComputerVisionCLI(config)

def add_to_terminal(envname: str='venv'):
    shell = os.environ.get('SHELL', '')
    if 'bash' in shell:
        profile_file = os.path.expanduser('~/.bashrc')
    elif 'zsh' in shell:
        profile_file = os.path.expanduser('~/.zshrc')
    else:
        print("Unsupported shell. Manual alias setup required.")
        return
    base_dir = get_base_dir()
    subprocess.check_call([sys.executable, '-m', 'venv', envname])
    pip_executable = os.path.join(base_dir, envname, 'bin', 'pip')
    req_file = get_requirements('computer_vision', as_list=False)
    subprocess.check_call([pip_executable, 'install', '-r', req_file])
    alias_cmd = "\nexport cv() {" + \
        f"source {os.path.join(base_dir, envname, 'bin', 'activate')} && " + \
        f"python {os.path.join(base_dir, 'custom_modules', 'computer_vision', 'cli.py')} $@ && " + \
        "deactivate }\n"
    with open(profile_file, 'a') as profile:
        profile.write(alias_cmd)

    print(f"Alias 'cv' created. Please restart the terminal or source the profile file.")


if __name__ == '__main__':
    config = get_default_config()
    app = ComputerVisionCLI(config)
    app.run()
