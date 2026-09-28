from .landmarks import Landmarks


class ConfidenceValidator:
    MIN_VISIBILITY = 0.5
    MIN_POSE_DETECTION_RATE = 0.70
    MIN_REQUIRED_JOINT_AVAILABILITY = 0.65
    MIN_ANGLE_CALCULABILITY_RATE = 0.65
    MAX_DROPOUT_RATE = 0.25
    MIN_PROCESSED_FRAMES = 15
    MIN_SIGNAL_RANGE = {
        "squat": 10.0,
        "bicep_curl": 20.0,
    }

    REQUIRED_POINTS = {
        "squat": {
            "left": [
                Landmarks.LEFT_SHOULDER,
                Landmarks.LEFT_HIP,
                Landmarks.LEFT_KNEE,
                Landmarks.LEFT_ANKLE,
            ],
            "right": [
                Landmarks.RIGHT_SHOULDER,
                Landmarks.RIGHT_HIP,
                Landmarks.RIGHT_KNEE,
                Landmarks.RIGHT_ANKLE,
            ],
        },
        "bicep_curl": {
            "left": [
                Landmarks.LEFT_SHOULDER,
                Landmarks.LEFT_ELBOW,
                Landmarks.LEFT_WRIST,
            ],
            "right": [
                Landmarks.RIGHT_SHOULDER,
                Landmarks.RIGHT_ELBOW,
                Landmarks.RIGHT_WRIST,
            ],
        },
    }

    ANGLE_POINTS = {
        "squat": {
            "left": (Landmarks.LEFT_HIP, Landmarks.LEFT_KNEE, Landmarks.LEFT_ANKLE),
            "right": (Landmarks.RIGHT_HIP, Landmarks.RIGHT_KNEE, Landmarks.RIGHT_ANKLE),
        },
        "bicep_curl": {
            "left": (Landmarks.LEFT_SHOULDER, Landmarks.LEFT_ELBOW, Landmarks.LEFT_WRIST),
            "right": (Landmarks.RIGHT_SHOULDER, Landmarks.RIGHT_ELBOW, Landmarks.RIGHT_WRIST),
        },
    }

    def __init__(self, exercise):
        self.exercise = exercise
        self.processed_frames = 0
        self.pose_detected_frames = 0
        self.required_available_frames = 0
        self.angle_calculable_frames = 0
        self.dropout_count = 0
        self.previous_required_available = None
        self.signal_values = []

    def _points_available(self, landmarks, points):
        return all(
            point in landmarks
            and landmarks[point].get("visibility", 0) >= self.MIN_VISIBILITY
            for point in points
        )

    def _side_angle(self, detector, landmarks, side):
        points = self.ANGLE_POINTS[self.exercise][side]
        if not self._points_available(landmarks, points):
            return None
        return detector.calculate_angle(landmarks, *points)

    def _percentile(self, values, percentage):
        values = sorted(values)
        if not values:
            return 0
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * percentage
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    def update(self, detector, landmarks):
        self.processed_frames += 1
        if landmarks:
            self.pose_detected_frames += 1

        required = self.REQUIRED_POINTS[self.exercise]
        left_available = self._points_available(landmarks, required["left"])
        right_available = self._points_available(landmarks, required["right"])
        required_available = left_available or right_available

        if required_available:
            self.required_available_frames += 1

        if self.previous_required_available is True and not required_available:
            self.dropout_count += 1
        self.previous_required_available = required_available

        angles = []
        for side in ("left", "right"):
            angle = self._side_angle(detector, landmarks, side)
            if angle is not None:
                angles.append(angle)

        if angles:
            self.angle_calculable_frames += 1
            self.signal_values.append(sum(angles) / len(angles))

    def _rate(self, numerator):
        if self.processed_frames == 0:
            return 0
        return numerator / self.processed_frames

    def _signal_range(self):
        if not self.signal_values:
            return 0
        p10 = self._percentile(self.signal_values, 0.10)
        p90 = self._percentile(self.signal_values, 0.90)
        return max(0, p90 - p10)

    def _message(self, reason_codes):
        if "TOO_FEW_FRAMES" in reason_codes:
            return "Analysis confidence is low because the video is too short. Please record a longer clip."
        if "NO_REPS_DETECTED" in reason_codes:
            return "Analysis confidence is low because no valid repetitions were detected. Please record again with the full movement visible."
        if "INSUFFICIENT_MOVEMENT_SIGNAL" in reason_codes:
            return "Analysis confidence is low because the movement signal was too small to analyse reliably. Please record again with a clearer full movement."
        if "LOW_REQUIRED_JOINT_AVAILABILITY" in reason_codes:
            if self.exercise == "squat":
                return "Analysis confidence is low because the full body was not clearly visible. Please record again with shoulders, hips, knees, and ankles in frame."
            return "Analysis confidence is low because the working arm was not clearly visible. Please record again with shoulders, elbows, and wrists in frame."
        if "LOW_POSE_DETECTION" in reason_codes:
            return "Analysis confidence is low because the body was not detected clearly enough. Please improve lighting, keep the camera stable, and record again."
        if "TRACKING_DROPOUT" in reason_codes:
            return "Analysis confidence is low because pose tracking dropped out repeatedly. Please keep the camera stable and the body visible."
        if "LOW_ANGLE_CALCULABILITY" in reason_codes:
            return "Analysis confidence is low because important joint angles could not be calculated reliably. Please record again from a clearer angle."
        return "Analysis confidence is low. Please record again with the required joints visible and the camera stable."

    def result(self, total_reps=0):
        pose_detection_rate = self._rate(self.pose_detected_frames)
        required_rate = self._rate(self.required_available_frames)
        angle_rate = self._rate(self.angle_calculable_frames)
        dropout_rate = self.dropout_count / max(self.processed_frames - 1, 1)
        signal_range = self._signal_range()

        reason_codes = []
        if self.processed_frames < self.MIN_PROCESSED_FRAMES:
            reason_codes.append("TOO_FEW_FRAMES")
        if pose_detection_rate < self.MIN_POSE_DETECTION_RATE:
            reason_codes.append("LOW_POSE_DETECTION")
        if required_rate < self.MIN_REQUIRED_JOINT_AVAILABILITY:
            reason_codes.append("LOW_REQUIRED_JOINT_AVAILABILITY")
        if angle_rate < self.MIN_ANGLE_CALCULABILITY_RATE:
            reason_codes.append("LOW_ANGLE_CALCULABILITY")
        if dropout_rate > self.MAX_DROPOUT_RATE:
            reason_codes.append("TRACKING_DROPOUT")
        if total_reps == 0:
            reason_codes.append("NO_REPS_DETECTED")
        if signal_range < self.MIN_SIGNAL_RANGE[self.exercise]:
            reason_codes.append("INSUFFICIENT_MOVEMENT_SIGNAL")

        status = "LOW" if reason_codes else "HIGH"
        high_message = "Analysis confidence is high. Repetition count and form feedback can be shown."
        return {
            "confidence_status": status,
            "usable_frame_rate": round(pose_detection_rate, 4),
            "required_joint_availability": round(required_rate, 4),
            "angle_calculability_rate": round(angle_rate, 4),
            "dropout_rate": round(dropout_rate, 4),
            "movement_signal_range": round(signal_range, 2),
            "reason_codes": reason_codes,
            "user_message": high_message if status == "HIGH" else self._message(reason_codes),
        }
