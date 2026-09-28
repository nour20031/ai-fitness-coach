def _count_issues(rep_history):
    issue_counts = {}
    for rep in rep_history:
        for issue in rep.get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    return issue_counts


def _rep_word(count):
    return "rep" if count == 1 else "reps"


def _issue_count_text(count):
    return f"{count} {_rep_word(count)}"


def _was_were(count):
    return "was" if count == 1 else "were"


def _no_reps_feedback(exercise_name):
    return (
        f"No complete {exercise_name} reps were detected. "
        "Make sure your full body or working joints are visible, then perform the movement slowly."
    )


def _generate_squat_feedback(summary):
    total_reps = summary.get("total_reps", 0)
    rep_history = summary.get("rep_history", [])
    issue_counts = summary.get("issue_counts") or _count_issues(rep_history)

    if total_reps == 0:
        return _no_reps_feedback("squat")

    good_reps = sum(1 for rep in rep_history if not rep.get("issues"))
    shallow_count = issue_counts.get("not_deep_enough", 0)
    forward_lean_count = issue_counts.get("forward_lean", 0)

    feedback = [
        f"You completed {total_reps} squat {_rep_word(total_reps)}.",
    ]

    if good_reps == total_reps:
        feedback.append("Your squat depth looked good across the detected reps.")
    elif good_reps > 0:
        feedback.append(
            f"{good_reps} {_rep_word(good_reps)} looked solid, but some reps need improvement."
        )
    else:
        feedback.append("Most detected reps need form improvement before increasing intensity.")

    if shallow_count:
        feedback.append(
            f"{_issue_count_text(shallow_count)} {_was_were(shallow_count)} shallow. "
            "Try lowering your hips a little more while keeping control."
        )

    if forward_lean_count:
        feedback.append(
            f"{_issue_count_text(forward_lean_count)} showed forward lean. "
            "Keep your chest lifted and brace your core as you descend."
        )

    if not issue_counts:
        feedback.append("Keep the same tempo and camera position for your next test.")
    else:
        feedback.append("Focus on one correction at a time, then retest with another short set.")

    return " ".join(feedback)


def _generate_bicep_curl_feedback(summary):
    total_reps = summary.get("total_reps", 0)
    rep_history = summary.get("rep_history", [])
    issue_counts = _count_issues(rep_history)

    if total_reps == 0:
        return _no_reps_feedback("bicep curl")

    good_reps = sum(1 for rep in rep_history if not rep.get("issues"))
    half_range_count = issue_counts.get("half_range_of_motion", 0)
    swing_count = issue_counts.get("body_swinging", 0)

    feedback = [
        f"You completed {total_reps} bicep curl {_rep_word(total_reps)}.",
    ]

    if good_reps == total_reps:
        feedback.append("Your curl range looked good and controlled across the detected reps.")
    elif good_reps > 0:
        feedback.append(
            f"{good_reps} {_rep_word(good_reps)} had good form, but some reps need adjustment."
        )
    else:
        feedback.append("Most detected reps need better control and range of motion.")

    if half_range_count:
        feedback.append(
            f"{_issue_count_text(half_range_count)} had half range of motion. "
            "Curl higher and close the elbow angle more before lowering the weight."
        )

    if swing_count:
        feedback.append(
            f"{_issue_count_text(swing_count)} showed body swinging. "
            "Keep your shoulders steady and lift mainly with your biceps."
        )

    if not issue_counts:
        feedback.append("Keep your elbows close to your body and continue using a controlled tempo.")
    else:
        feedback.append("Use a lighter weight or slower tempo until the movement stays controlled.")

    return " ".join(feedback)


def generate_feedback(exercise, summary):
    exercise_key = exercise.lower().strip().replace(" ", "_")

    if exercise_key == "squat":
        return _generate_squat_feedback(summary)

    if exercise_key in ("bicep_curl", "biceps", "curl"):
        return _generate_bicep_curl_feedback(summary)

    return "Feedback is not available for this exercise yet."
