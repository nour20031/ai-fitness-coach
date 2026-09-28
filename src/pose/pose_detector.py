import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self, mode=False, smooth=True, detection_conf=0.5, tracking_conf=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.tracking_conf
        )

    def find_pose(self, frame, draw=True):
        """Detect pose landmarks in a frame"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame

    def get_landmarks(self, frame):
        """Extract landmark positions as a dictionary"""
        landmarks = {}

        if self.results.pose_landmarks:
            h, w, _ = frame.shape
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                landmarks[idx] = {
                    'x': int(lm.x * w),
                    'y': int(lm.y * h),
                    'visibility': lm.visibility
                }
        return landmarks

    def calculate_angle(self, landmarks, p1, p2, p3):
        """
        Calculate angle between three landmarks.
        p2 is the middle point (the joint i am  measuring it ).
        Example: p1=hip, p2=knee, p3=ankle gives knee angle
        """
        if not all(k in landmarks for k in [p1, p2, p3]):
            return None

        a = np.array([landmarks[p1]['x'], landmarks[p1]['y']])
        b = np.array([landmarks[p2]['x'], landmarks[p2]['y']])
        c = np.array([landmarks[p3]['x'], landmarks[p3]['y']])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return round(angle, 2)

    def draw_angle(self, frame, landmarks, point, angle):
        """Display angle value on frame at the joint location"""
        if point in landmarks and angle is not None:
            x = landmarks[point]['x']
            y = landmarks[point]['y']
            cv2.putText(
                frame, str(angle),
                (x - 30, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2
            )
        return frame