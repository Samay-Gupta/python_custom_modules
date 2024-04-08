import cv2
import imutils
import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils import paths

class FaceDetector:
    def classify(self, frame, min_conf=0.5):
        """
        Method to detect faces in the given frame. This method should be implemented 
        in subclasses.

        :param frame: The image frame to detect faces.
        :param min_conf: Minimum confidence threshold for detecting faces.
        :return: A list of detections, each a dictionary with bounding box and confidence.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def draw(self, frame, detections, color: tuple = (0, 0, 255)):
        """
        Draws bounding boxes on the frame for each detected face.

        :param frame: The original image frame.
        :param detections: A list of detections to draw.
        :param color: Color tuple for the bounding box.
        """
        for detection in detections:
            start_x, start_y, end_x, end_y = detection["bounding_box"]
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
        return frame

class CaffeFaceDetector(FaceDetector):
    def __init__(self, model_dir):
        """
        Initialize the CaffeFaceDetector class with the model directory.

        :param model_dir: Directory where the Caffe model files are stored.
        """
        caffe_file_path = os.path.join(model_dir, "res_ssd_300x300.caffemodel")
        prototxt_file_path = os.path.join(model_dir, "deploy.prototxt")
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_file_path, caffe_file_path)

    def classify(self, frame, min_conf=0.5):
        """
        Detects faces in the given frame using the Caffe model.

        :param frame: The image frame to detect faces.
        :param min_conf: Minimum confidence threshold for detecting faces.
        :return: A list of detections, each a dictionary with bounding box and confidence.
        """
        image_height, image_width = frame.shape[:2]
        box_factor = np.array([image_width, image_height, image_width, image_height])
        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                           (300, 300), (104.0, 177.0, 123.0))
        self.detector.setInput(image_blob)
        detections = self.detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >= min_conf:
                bounding_box = (detections[0, 0, i, 3:7] * box_factor).astype("int")
                faces.append({
                    "bounding_box": bounding_box,
                    "confidence": confidence
                })
        return faces

class FaceClassifier:
    def classify(self, frame):
        """
        Classify faces in the given frame. This method should be implemented in subclasses.

        :param frame: The image frame to classify faces.
        :return: Classification results.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def train(self, training_images_dir):
        """
        Train the face classifier using images from the specified directory. This method 
        should be implemented in subclasses.

        :param training_images_dir: Directory containing training images.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def draw(self, frame, classifications, color: tuple = (0, 255, 0)):
        """
        Draws bounding boxes and labels on the frame for each classified face.

        :param frame: The original image frame.
        :param classifications: A list of classifications to draw.
        :param color: Color tuple for the bounding box and text.
        """
        for classification in classifications:
            start_x, start_y, end_x, end_y = classification["bounding_box"]
            label = classification["label"]
            confidence = classification["confidence"]
            text = f"{label}: {confidence:.2f}%"
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
            cv2.putText(frame, text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


class HaarFaceClassifier(FaceClassifier):
    def __init__(self, model_dir, face_detector):
        """
        Initializes the HaarFaceClassifier with a model directory and a Haar cascade face detector.

        :param model_dir: Directory where the HaarFaceClassifier's model files are stored.
        :param face_detector: An instance of a Haar cascade face detector.
        """
        self.model_dir = model_dir
        self.face_detector = face_detector
        self.data_embedder = cv2.dnn.readNetFromTorch(os.path.join(model_dir, "openface_nn4.small2.v1.t7"))
        self.label_encoder = pickle.load(open(os.path.join(model_dir, "encoder.pickle"), "rb"))
        self.classifier = pickle.load(open(os.path.join(model_dir, "recognizer.pickle"), "rb"))

    def classify(self, frame):
        """
        Classifies faces in the given frame using the Haar cascade face detector and the classifier.

        :param frame: The image frame to classify faces.
        :return: Classification results as a list of dictionaries containing bounding box, confidence, and label.
        """
        classifications = []
        face_detections = self.face_detector.classify(frame)
        for detection in face_detections:
            (start_x, start_y, width, height) = detection["bounding_box"]
            face_image = frame[start_y:start_y + height, start_x:start_x + width]
            if min(face_image.shape[:2]) < 20:
                continue
            face_image_blob = cv2.dnn.blobFromImage(face_image, 1.0/255, (96, 96), 
                                                    (0, 0, 0), swapRB=True, crop=False)
            self.data_embedder.setInput(face_image_blob)
            visual_encoder = self.data_embedder.forward()
            predictions = self.classifier.predict_proba(visual_encoder)[0]
            label = self.label_encoder.classes_[np.argmax(predictions)]
            confidence = detection["confidence"] * 100
            classifications.append({
                "bounding_box": detection["bounding_box"],
                "confidence": confidence,
                "label": label,
            })
        return classifications

    def train(self, training_images_dir):
        """
        Trains the HaarFaceClassifier using images from the specified directory.

        :param training_images_dir: Directory containing training images.
        """
        recognizer_file_path = os.path.join(self.model_dir, "recognizer.pickle")
        pretrained_embeddings_path = os.path.join(self.model_dir, "embeddings.pickle")
        encoder_file_path = os.path.join(self.model_dir, "encoder.pickle")

        image_files = list(paths.list_images(training_images_dir))
        training_data = []
        training_labels = []
        for image_path in image_files:
            image_label = image_path.split(os.path.sep)[-2]
            img = cv2.imread(image_path)
            image_file = imutils.resize(img, width=600)
            detections = self.face_detector.classify(image_file)
            for detection in detections:
                x0, y0, w, h = detection["bounding_box"]
                x1, y1 = x0 + w, y0 + h
                face_image = image_file[y0:y1, x0:x1]
                if min(face_image.shape[:2]) < 20:
                    continue
                face_image_blob = cv2.dnn.blobFromImage(face_image, 1.0/255, (96, 96), 
                                                        (0, 0, 0), swapRB=True, crop=False)
                self.data_embedder.setInput(face_image_blob)
                visual_encoder = self.data_embedder.forward()
                image_data = visual_encoder.flatten()
                training_data.append(image_data)
                training_labels.append(image_label)

        embedding_data = {"data": training_data, "labels": training_labels}
        with open(pretrained_embeddings_path, 'wb') as file_object:
            file_object.write(pickle.dumps(embedding_data))

        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(training_labels)
        clf = SVC(C=1.0, kernel="linear", probability=True)
        clf.fit(training_data, train_labels)
        with open(recognizer_file_path, 'wb') as file_object:
            file_object.write(pickle.dumps(clf))
        with open(encoder_file_path, 'wb') as file_object:
            file_object.write(pickle.dumps(label_encoder))

        self.label_encoder = label_encoder
        self.classifier = clf
