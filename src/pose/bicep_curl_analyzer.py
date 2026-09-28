import cv2

from .landmarks import Landmarks


class BicepCurlAnalyzer:
    def __init__(self, require_ready=True, verbose=True):
        # Bicep curl angles: extended arm is a high angle, curled arm is a low angle.
        self.ELBOW_DOWN_TRIGGER = 125
        self.ELBOW_UP_TRIGGER = 115
        self.HALF_REP_THRESHOLD = 55
        self.BODY_SWING_THRESHOLD_RATIO = 0.06
        self.MIN_VISIBILITY = 0.5
        self.STABLE_FRAMES_REQUIRED = 4
        self.COOLDOWN_FRAMES = 8
        self.READY_FRAMES_REQUIRED = 15

        self.rep_count = 0
        self.stage = None
        self.require_ready = require_ready
        self.verbose = verbose
        self.is_ready = not require_ready
        self.ready_frame_count = 0
        self.rep_history = []
        self.max_elbow_angle = 0
        self.min_elbow_angle = 180
        self.start_shoulder_y = None
        self.max_shoulder_deviation = 0
        self.down_frame_count = 0
        self.up_frame_count = 0
        self.cooldown = 0

    def _check_visibility(self, landmarks, *points):
        for point in points:
            if point not in landmarks:
                return False
            if landmarks[point]["visibility"] < self.MIN_VISIBILITY:
                return False
        return True

    def _get_visible_arm_angles(self, detector, landmarks):
        visible_arms = []
        arm_points = [
            ("left", Landmarks.LEFT_SHOULDER, Landmarks.LEFT_ELBOW, Landmarks.LEFT_WRIST),
            ("right", Landmarks.RIGHT_SHOULDER, Landmarks.RIGHT_ELBOW, Landmarks.RIGHT_WRIST),
        ]

        for side, shoulder, elbow, wrist in arm_points:
            if not self._check_visibility(landmarks, shoulder, elbow, wrist):
                continue

            angle = detector.calculate_angle(landmarks, shoulder, elbow, wrist)
            if angle is None:
                continue

            visible_arms.append({
                "side": side,
                "angle": angle,
                "elbow": elbow,
            })

        return visible_arms

    def _detect_body_swing(self, landmarks):
        shoulder_values = []
        for shoulder in [Landmarks.LEFT_SHOULDER, Landmarks.RIGHT_SHOULDER]:
            if shoulder in landmarks and landmarks[shoulder]["visibility"] >= self.MIN_VISIBILITY:
                shoulder_values.append(landmarks[shoulder]["y"])

        if not shoulder_values:
            return 0

        shoulder_y = sum(shoulder_values) / len(shoulder_values)
        if self.start_shoulder_y is None:
            self.start_shoulder_y = shoulder_y
            return 0

        return abs(shoulder_y - self.start_shoulder_y)

    def _finish_rep(self, frame):
        self.stage = "down"
        self.rep_count += 1
        rep_issues = []

        if self.min_elbow_angle > self.HALF_REP_THRESHOLD:
            rep_issues.append("half_range_of_motion")

        swing_threshold = frame.shape[0] * self.BODY_SWING_THRESHOLD_RATIO
        if self.max_shoulder_deviation > swing_threshold:
            rep_issues.append("body_swinging")

        self.rep_history.append({
            "rep": self.rep_count,
            "min_elbow_angle": round(self.min_elbow_angle, 1),
            "max_elbow_angle": round(self.max_elbow_angle, 1),
            "max_shoulder_deviation": round(self.max_shoulder_deviation, 1),
            "issues": rep_issues,
        })

        if self.verbose:
            print(f"\n--- Rep {self.rep_count} Complete ---")
            print(f"  Min elbow angle (curled): {round(self.min_elbow_angle, 1)} deg")
            print(f"  Max elbow angle (extended): {round(self.max_elbow_angle, 1)} deg")
            print(f"  Shoulder deviation: {round(self.max_shoulder_deviation, 1)}px")
            if rep_issues:
                print(f"  Issues: {rep_issues}")
            else:
                print("  Form: GOOD")

        self.min_elbow_angle = 180
        self.max_elbow_angle = 0
        self.max_shoulder_deviation = 0
        self.start_shoulder_y = None
        return rep_issues

    def _reset_current_rep_tracking(self):
        self.min_elbow_angle = 180
        self.max_elbow_angle = 0
        self.max_shoulder_deviation = 0
        self.start_shoulder_y = None
        self.down_frame_count = 0
        self.up_frame_count = 0

    def analyze(self, detector, landmarks, frame):
        issues = []
        angles = {}

        visible_arms = self._get_visible_arm_angles(detector, landmarks)
        if not visible_arms:
            cv2.putText(frame, "Move into frame", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Reps: {self.rep_count}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            return {
                "exercise": "bicep_curl",
                "reps": self.rep_count,
                "stage": self.stage,
                "angles": {},
                "issues": [],
            }

        # Average both arms when possible. Using only the most flexed arm is too
        # sensitive because one noisy/hidden arm can create fake reps.
        elbow_angle = sum(arm["angle"] for arm in visible_arms) / len(visible_arms)
        active_arm = min(visible_arms, key=lambda arm: arm["angle"])
        angles["elbow"] = round(elbow_angle, 1)
        angles["side"] = "both" if len(visible_arms) == 2 else active_arm["side"]

        was_ready = self.is_ready
        if not self.is_ready:
            if elbow_angle > self.ELBOW_DOWN_TRIGGER:
                self.ready_frame_count += 1
            else:
                self.ready_frame_count = 0

            if self.ready_frame_count >= self.READY_FRAMES_REQUIRED:
                self.is_ready = True
                self.stage = "down"
                self._reset_current_rep_tracking()

        if self.is_ready and 10 < elbow_angle < 180:
            if self.cooldown > 0:
                self.cooldown -= 1

            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            self.max_elbow_angle = max(self.max_elbow_angle, elbow_angle)

            shoulder_deviation = self._detect_body_swing(landmarks)
            self.max_shoulder_deviation = max(
                self.max_shoulder_deviation,
                shoulder_deviation,
            )

            if elbow_angle > self.ELBOW_DOWN_TRIGGER:
                self.down_frame_count += 1
                self.up_frame_count = 0
            elif elbow_angle < self.ELBOW_UP_TRIGGER:
                self.up_frame_count += 1
                self.down_frame_count = 0
            else:
                self.down_frame_count = 0
                self.up_frame_count = 0

            # Full rep: stable extended/down -> stable curled/up -> stable
            # extended/down. Stable-frame checks avoid counting random jitter.
            if (
                self.stage == "up"
                and self.down_frame_count >= self.STABLE_FRAMES_REQUIRED
                and self.cooldown == 0
            ):
                issues = self._finish_rep(frame)
                self.cooldown = self.COOLDOWN_FRAMES
                self.down_frame_count = 0
                self.up_frame_count = 0
            elif self.down_frame_count >= self.STABLE_FRAMES_REQUIRED:
                self.stage = "down"
            elif self.up_frame_count >= self.STABLE_FRAMES_REQUIRED and self.stage in (None, "down"):
                self.stage = "up"

        for arm in visible_arms:
            frame = detector.draw_angle(frame, landmarks,
                                        arm["elbow"], round(arm["angle"], 1))

        cv2.putText(frame, f"Reps: {self.rep_count}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Stage: {self.stage}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if self.require_ready and not self.is_ready:
            cv2.putText(frame, "Hold arms extended to start",
                        (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        color = (0, 255, 0) if angles["elbow"] < self.HALF_REP_THRESHOLD else (255, 255, 0)
        cv2.putText(frame, f"Elbow: {angles['elbow']}",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Arm: {angles['side']}",
                    (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return {
            "exercise": "bicep_curl",
            "reps": self.rep_count,
            "stage": self.stage,
            "angles": angles,
            "issues": issues,
        }

    def get_session_summary(self):
        issue_counts = {}
        for rep in self.rep_history:
            for issue in rep.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        half_range_reps = issue_counts.get("half_range_of_motion", 0)
        if self.rep_count == 0:
            predicted_form_label = "unknown"
            form_label_reason = "no_complete_curl_reps_detected"
        elif half_range_reps > self.rep_count / 2:
            predicted_form_label = "half_range"
            form_label_reason = "majority_of_detected_reps_above_curl_depth_threshold"
        else:
            predicted_form_label = "correct"
            form_label_reason = "majority_of_detected_reps_reached_full_range_threshold"

        return {
            "total_reps": self.rep_count,
            "issue_counts": issue_counts,
            "rep_history": self.rep_history,
            "predicted_form_label": predicted_form_label,
            "form_label_reason": form_label_reason,
            "thresholds": {
                "elbow_down_trigger": self.ELBOW_DOWN_TRIGGER,
                "elbow_up_trigger": self.ELBOW_UP_TRIGGER,
                "half_rep_threshold": self.HALF_REP_THRESHOLD,
                "stable_frames_required": self.STABLE_FRAMES_REQUIRED,
                "cooldown_frames": self.COOLDOWN_FRAMES,
            },
        }

    def reset(self):
        self.rep_count = 0
        self.stage = None
        self.is_ready = not self.require_ready
        self.ready_frame_count = 0
        self.rep_history = []
        self.min_elbow_angle = 180
        self.max_elbow_angle = 0
        self.max_shoulder_deviation = 0
        self.start_shoulder_y = None
        self.down_frame_count = 0
        self.up_frame_count = 0
        self.cooldown = 0
