import cv2

from .landmarks import Landmarks


class SquatAnalyzer:
    def __init__(self, verbose=True):
        self.KNEE_BAD_DEPTH = 115
        self.KNEE_DOWN_TRIGGER = 165
        self.KNEE_UP_TRIGGER = 165
        self.TORSO_FORWARD_LEAN = 25
        self.MIN_VISIBILITY = 0.5
        self.MIN_DOWN_FRAMES = 3
        self.verbose = verbose

        self.rep_count = 0
        self.stage = None
        self.rep_history = []
        self.min_knee_angle = 180
        self.current_rep_issues = set()
        self.down_frame_count = 0

    def _check_visibility(self, landmarks, *points):
        for point in points:
            if point not in landmarks:
                return False
            if landmarks[point]["visibility"] < self.MIN_VISIBILITY:
                return False
        return True

    def _get_visible_leg_angles(self, detector, landmarks):
        legs = []
        leg_points = [
            ("left", Landmarks.LEFT_HIP, Landmarks.LEFT_KNEE, Landmarks.LEFT_ANKLE),
            ("right", Landmarks.RIGHT_HIP, Landmarks.RIGHT_KNEE, Landmarks.RIGHT_ANKLE),
        ]

        for side, hip, knee, ankle in leg_points:
            if not self._check_visibility(landmarks, hip, knee, ankle):
                continue

            angle = detector.calculate_angle(landmarks, hip, knee, ankle)
            if angle is None:
                continue

            legs.append({
                "side": side,
                "angle": angle,
                "knee": knee,
                "hip": hip,
            })

        return legs

    def _get_torso_angle(self, detector, landmarks, preferred_side):
        side_points = {
            "left": (Landmarks.LEFT_SHOULDER, Landmarks.LEFT_HIP, Landmarks.LEFT_KNEE),
            "right": (Landmarks.RIGHT_SHOULDER, Landmarks.RIGHT_HIP, Landmarks.RIGHT_KNEE),
        }

        ordered_sides = [preferred_side, "left" if preferred_side == "right" else "right"]
        for side in ordered_sides:
            shoulder, hip, knee = side_points[side]
            if not self._check_visibility(landmarks, shoulder, hip, knee):
                continue

            angle = detector.calculate_angle(landmarks, shoulder, hip, knee)
            if angle is not None:
                return angle, hip

        return None, None

    def _finish_rep(self):
        self.rep_count += 1
        self.stage = "up"

        if self.min_knee_angle > self.KNEE_BAD_DEPTH:
            self.current_rep_issues.add("not_deep_enough")

        rep_issues = sorted(self.current_rep_issues)

        self.rep_history.append({
            "rep": self.rep_count,
            "min_knee_angle": round(self.min_knee_angle, 1),
            "issues": rep_issues,
        })

        if self.verbose:
            print(f"\n--- Rep {self.rep_count} Complete ---")
            print(f"  Deepest knee angle: {round(self.min_knee_angle, 1)} deg")
            print(f"  Issues: {rep_issues if rep_issues else 'None'}")

        self.min_knee_angle = 180
        self.current_rep_issues = set()
        self.down_frame_count = 0

        return rep_issues

    def finalize(self):
        """Count a strong final down-position if the clip ends before the user stands fully up."""
        if (
            self.stage == "down"
            and self.down_frame_count >= self.MIN_DOWN_FRAMES
            and self.min_knee_angle < self.KNEE_DOWN_TRIGGER
        ):
            return self._finish_rep()

        return []

    def analyze(self, detector, landmarks, frame):
        issues = []
        angles = {}

        visible_legs = self._get_visible_leg_angles(detector, landmarks)

        if not visible_legs:
            cv2.putText(frame, "Move into frame", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Reps: {self.rep_count}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            return {
                "exercise": "squat",
                "reps": self.rep_count,
                "stage": self.stage,
                "angles": {},
                "issues": [],
            }

        knee_angle = sum(leg["angle"] for leg in visible_legs) / len(visible_legs)
        active_leg = min(visible_legs, key=lambda leg: leg["angle"])
        angles["knee"] = round(knee_angle, 1)
        angles["side"] = "both" if len(visible_legs) == 2 else active_leg["side"]

        if 10 < knee_angle < 180:
            self.min_knee_angle = min(self.min_knee_angle, knee_angle)

            if knee_angle < self.KNEE_DOWN_TRIGGER:
                self.stage = "down"
                self.down_frame_count += 1

            if (
                self.stage == "down"
                and self.down_frame_count >= self.MIN_DOWN_FRAMES
                and knee_angle > self.KNEE_UP_TRIGGER
            ):
                issues = self._finish_rep()

        for leg in visible_legs:
            frame = detector.draw_angle(
                frame,
                landmarks,
                leg["knee"],
                round(leg["angle"], 1),
            )

        torso_angle, torso_point = self._get_torso_angle(
            detector,
            landmarks,
            active_leg["side"],
        )

        if torso_angle is not None and torso_angle > 5:
            angles["torso"] = round(torso_angle, 1)
            frame = detector.draw_angle(frame, landmarks, torso_point, round(torso_angle, 1))

            if self.stage == "down" and torso_angle < self.TORSO_FORWARD_LEAN:
                self.current_rep_issues.add("forward_lean")

        cv2.putText(frame, f"Reps: {self.rep_count}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Stage: {self.stage}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if "knee" in angles:
            cv2.putText(frame, f"Knee: {angles['knee']}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)
            cv2.putText(frame, f"Leg: {angles['side']}",
                        (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

        return {
            "exercise": "squat",
            "reps": self.rep_count,
            "stage": self.stage,
            "angles": angles,
            "issues": issues,
        }

    def get_session_summary(self):
        issue_counts = {}

        for rep in self.rep_history:
            for issue in rep["issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        shallow_reps = issue_counts.get("not_deep_enough", 0)
        if self.rep_count == 0:
            predicted_form_label = "unknown"
            form_label_reason = "no_complete_squat_reps_detected"
        elif shallow_reps > self.rep_count / 2:
            predicted_form_label = "shallow"
            form_label_reason = "majority_of_detected_reps_above_depth_threshold"
        else:
            predicted_form_label = "correct"
            form_label_reason = "majority_of_detected_reps_reached_depth_threshold"

        return {
            "total_reps": self.rep_count,
            "issue_counts": issue_counts,
            "rep_history": self.rep_history,
            "predicted_form_label": predicted_form_label,
            "form_label_reason": form_label_reason,
            "thresholds": {
                "knee_bad_depth": self.KNEE_BAD_DEPTH,
                "knee_down_trigger": self.KNEE_DOWN_TRIGGER,
                "knee_up_trigger": self.KNEE_UP_TRIGGER,
                "minimum_down_frames": self.MIN_DOWN_FRAMES,
            },
        }

    def reset(self):
        self.rep_count = 0
        self.stage = None
        self.rep_history = []
        self.min_knee_angle = 180
        self.current_rep_issues = set()
        self.down_frame_count = 0
