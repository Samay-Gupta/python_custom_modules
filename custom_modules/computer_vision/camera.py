import cv2
import time
from threading import Thread
from typing import Callable, Tuple, Union
import numpy as np

class Camera:
    """
    A class representing a camera.

    Attributes:
        stream (cv2.VideoCapture): The video capture stream.
        is_recording (bool): Indicates if the camera is currently recording.
        frame (numpy.ndarray): The current frame captured by the camera.
        thread (Thread): The thread used for recording frames.
        callback (Callable[[np.ndarray], None]): The callback function to be called for each frame.

    Methods:
        set_callback(callback): Sets the callback function for each frame.
        get_frame(): Returns the current frame captured by the camera.
        get_size(): Returns the size of the captured frames.
        start_recording(): Starts recording frames from the camera.
        stop_recording(): Stops recording frames from the camera.
        get_single_frame(src): Captures a single frame from the camera.

    """

    def __init__(self, src: Union[int, str] = 0) -> None:
        """
        Initializes a new instance of the Camera class.

        Args:
            src (int or str): The source of the camera stream. Defaults to 0 (the default camera).

        """
        self.stream = cv2.VideoCapture(src)
        self.is_recording = False
        self.frame = None
        self.thread = None
        self.callback = None

    def set_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Sets the callback function to be called for each frame.

        Args:
            callback (function): The callback function to be called for each frame.

        """
        self.callback = callback

    def __record(self) -> None:
        """
        Private method that continuously reads frames from the camera stream and updates the current frame.

        """
        while self.is_recording:
            ret, frame = self.stream.read()
            if ret: 
                self.frame = frame
                if self.callback is not None:
                    self.callback(frame)

    def __wait_for_frame(self) -> None:
        """
        Private method that waits for the first frame to be captured.

        """
        while self.frame is None:
            time.sleep(0.1)
        time.sleep(0.5)

    def get_frame(self) -> np.ndarray:
        """
        Returns the current frame captured by the camera.

        Returns:
            numpy.ndarray: The current frame captured by the camera.

        """
        return self.frame

    def get_size(self) -> Tuple[int, int]:
        """
        Returns the size of the captured frames.

        Returns:
            tuple: A tuple containing the width and height of the captured frames.

        """
        return (
            int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    def start_recording(self) -> None:
        """
        Starts recording frames from the camera.

        """
        self.is_recording = True
        self.thread = Thread(target=self.__record, args=())
        self.thread.start()
        self.__wait_for_frame()

    def stop_recording(self) -> None:
        """
        Stops recording frames from the camera.

        """
        self.is_recording = False
        if self.stream.isOpened():
            self.stream.release()
        if self.thread is not None:
            self.thread.join()
        self.thread = None
    
    @staticmethod
    def get_single_frame(src: Union[int, str] = 0) -> np.ndarray:
        """
        Captures a single frame from the camera.

        Args:
            src (int or str): The source of the camera stream. Defaults to 0 (the default camera).

        Returns:
            numpy.ndarray: The captured frame.

        """
        camera = Camera(src=src)
        camera.start_recording()
        frame = camera.get_frame()
        camera.stop_recording()
        return frame
