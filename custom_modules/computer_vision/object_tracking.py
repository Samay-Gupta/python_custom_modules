from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import imutils
import dlib
import cv2


class TrackedObject:
    CTR = 0

    def __init__(self, object_id, centroid):
        self.object_id = object_id
        self.centroids = [centroid]
        self.drn = 0
        self.global_id = TrackedObject.CTR
        TrackedObject.CTR += 1

class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (c_x, c_y)
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

    
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

class BoundaryTracker:
    def __init__(self, reset_interval=30, boundary=[300, 0, 300, 500], scaled_width=500):
        self.trackers = []
        self.ct = CentroidTracker(40)
        self.trackable_objects = {}
        self.frame_count = 0
        self.reset_interval = reset_interval
        self.objects_crossed = 0
        self.boundary = boundary
        self.scaled_width = scaled_width

    def get_objects_crossed(self):
        return int(self.objects_crossed)

    def update(self, frame, object_rects):
        self.trackers = []
        for rect in object_rects:
            tracker = dlib.correlation_tracker()
            tracker.start_track(frame, dlib.rectangle(*rect))
            self.trackers.append(tracker)
        
    def track_objects(self, frame):
        rects = []
        for tracker in self.trackers:
            tracker.update(frame)
            pos = tracker.get_position()
            rects.append((int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())))
        return rects

    def count_objects(self, objects):
        for object_id, centroid in objects.items():
            to = self.trackable_objects.get(object_id, None)
            if to is None:
                to = TrackedObject(object_id, centroid)
            else:
                x, y = zip(*to.centroids)
                to.centroids.append(centroid)
                bound_start = np.array( self.boundary[:2])
                bound_end = np.array( self.boundary[2:])
                obj_drn = np.array([centroid[0] - np.mean(x), centroid[1] - np.mean(y)])
                cur_drn = np.sign(np.cross(bound_end - bound_start, obj_drn))
                crossed = False
                if len(to.centroids) > 1:
                    last_position = np.array(to.centroids[-2])
                    cross_last = np.cross(bound_end - bound_start, last_position - bound_start)
                    cross_current = np.cross(bound_end - bound_start, centroid - bound_start)
                    crossed = np.sign(cross_last) != np.sign(cross_current)
                if cur_drn != to.drn and cur_drn != 0 and crossed:   
                    self.objects_crossed += cur_drn
                    to.drn = cur_drn
            self.trackable_objects[object_id] = to

    def process_frame(self, frame, get_objects=None):
        ref = frame.copy()
        frame = imutils.resize(frame, width=self.scaled_width)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = []
        if self.frame_count % self.reset_interval == 0 and get_objects is not None:
            object_rects = get_objects(ref)
            self.update(frame, object_rects)
        else:
            rects = self.track_objects(frame)
        objects = self.ct.update(rects)
        self.count_objects(objects)
        self.frame_count += 1

    def draw(self, frame):
        cv2.line(frame,  self.boundary[:2],  self.boundary[2:], (255, 0, 0), 2)
        for object_id, to in self.trackable_objects.items():
            centroid = to.centroids[-1]
            text = f"ID {to.global_id} {to.drn}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        return frame
