from barcode.writer import ImageWriter
from pyzbar.pyzbar import ZBarSymbol
from barcode import EAN13
from pyzbar import pyzbar
import qrcode

class QR_Code:
    @staticmethod
    def generate(value: str, output: str = 'qr_code.png', version: int = 1, box_size: int = 20, border: int = 4) -> None:
        """
        Generates a QR code image from a given value.

        :param value: The data to be encoded in the QR code.
        :param output: The file name or path for the output image.
        :param version: The version of the QR code defines the size of the grid.
        :param box_size: The size of each box in the QR code grid.
        :param border: The width of the border around the QR code.
        """
        qr = qrcode.QRCode(
            version=version,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=box_size,
            border=border,
        )
        qr.add_data(value)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(output)

    @staticmethod
    def scan(image) -> list:
        """
        Scans an image and extracts QR code data.

        :param image: The image containing the QR code.
        :return: The decoded data from the QR code.
        """
        return pyzbar.decode(image, symbols=[ZBarSymbol.QRCODE])


class Barcode:
    @staticmethod
    def generate(value: str, output: str = 'barcode.png') -> None:
        """
        Generates a barcode image from a given value.

        :param value: The data to be encoded in the barcode.
        :param output: The file name or path for the output image.
        """
        with open(output, 'wb') as f:
            EAN13(value, writer=ImageWriter()).write(f)

    @staticmethod
    def scan(image) -> list:
        """
        Scans an image and extracts barcode data.

        :param image: The image containing the barcode.
        :return: The decoded data from the barcode.
        """
        return pyzbar.decode(image, symbols=[ZBarSymbol.EAN13])

class CodeScanner:
    @staticmethod
    def scan(image, code_type: str) -> list:
        """
        Scans an image and extracts data from either a QR code or a barcode.

        :param image: The image containing the code.
        :param code_type: The type of code to scan, either 'qr' or 'barcode'.
        :return: The decoded data from the code.
        """
        if code_type == 'qr':
            return QR_Code.scan(image)
        elif code_type == 'barcode':
            return Barcode.scan(image)
        else:
            raise ValueError(f"Invalid code type: {code_type}")

    @staticmethod
    def generate(value: str, code_type: str, output: str) -> None:
        """
        Generates a code image from a given value.

        :param value: The data to be encoded in the code.
        :param code_type: The type of code to generate, either 'qr' or 'barcode'.
        :param output: The file name or path for the output image.
        """
        if code_type == 'qr':
            QR_Code.generate(value, output)
        elif code_type == 'barcode':
            Barcode.generate(value, output)
        else:
            raise ValueError(f"Invalid code type: {code_type}")