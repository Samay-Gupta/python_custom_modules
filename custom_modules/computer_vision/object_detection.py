import numpy as np
import cv2
import os

class ObjectDetector:
    def __init__(self, model_dir):
        """
        Initializes the object detection model.

        :param model_dir: Directory where the YOLOv4 model files are stored.
        """
        model_cfg = os.path.join(model_dir, "yolov4.cfg")
        model_weights = os.path.join(model_dir, "yolov4.weights")
        model_labels = os.path.join(model_dir, "yolov4.names")

        self.model = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        ln = self.model.getLayerNames()
        self.output_layers = [ln[i - 1] for i in self.model.getUnconnectedOutLayers()]
        self.labels = self.read_labels(model_labels)

    def read_labels(self, label_path):
        with open(label_path) as file:
            return file.read().strip().split("\n")

    def classify(self, frame, min_conf=0.5, min_thresh=0.3):
        """
        Classifies objects in a frame using the YOLOv4 model.

        :param frame: The image frame to classify.
        :param min_conf: Minimum confidence threshold for detection.
        :param min_thresh: Threshold to filter weak detections using non-maxima suppression.
        :return: A list of detections, each containing a bounding box, confidence level, and label.
        """
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        layer_outputs = self.model.forward(self.output_layers)

        return self.process_detections(frame, layer_outputs, min_conf, min_thresh)

    def classify_from_camera(self, src=0, min_conf=0.5, min_thresh=0.3):
        """
        Classifies objects in a frame captured from the camera.

        :param min_conf: Minimum confidence threshold for detection.
        :param min_thresh: Threshold to filter weak detections using non-maxima suppression.
        :return: A list of detections, each containing a bounding box, confidence level, and label.
        """
        cap = cv2.VideoCapture(src)
        ret, frame = cap.read()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.classify_and_draw(frame, min_conf, min_thresh)
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    def classify_and_draw(self, frame, min_conf=0.5, min_thresh=0.3):
        """
        Classifies objects in a frame using the YOLOv4 model.

        :param frame: The image frame to classify.
        :param min_conf: Minimum confidence threshold for detection.
        :param min_thresh: Threshold to filter weak detections using non-maxima suppression.
        :return: A list of detections, each containing a bounding box, confidence level, and label.
        """
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        layer_outputs = self.model.forward(self.output_layers)
        H, W = frame.shape[:2]
        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > min_conf:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, min_thresh)
        if len(idxs) == 0:
            return frame
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{self.labels[class_ids[i]]}: {confidences[i]:.2f}%"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
        

    def process_detections(self, frame, layer_outputs, min_conf, min_thresh):
        H, W = frame.shape[:2]
        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > min_conf:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, min_thresh)
        if len(idxs) == 0:
            return []
        return [{
            "bounding_box": boxes[i],
            "confidence": confidences[i],
            "label": self.labels[class_ids[i]]
        } for i in idxs.flatten()]

    def draw(self, frame, detections):
        """
        Draws bounding boxes and labels on the frame for each detection.

        :param frame: The original image frame.
        :param detections: A list of detections to draw.
        :return: The image frame with detections drawn.
        """
        for det in detections:
            x, y, w, h = det["bounding_box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{det['label']}: {det['confidence']:.2f}%"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
