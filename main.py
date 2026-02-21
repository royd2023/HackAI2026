import cv2
import mediapipe as mp
import math
from collections import deque

cap = cv2.VideoCapture(1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """
    Interior angle at point b using the dot product of vectors ba and bc.
    Returns a value in [0, 180] degrees, which is geometrically correct
    and avoids the [-360, 360] range issue of the atan2 subtraction method.
    """
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
    if mag < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def to_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))


# Smooth raw angles over the last N frames to eliminate per-frame jitter.
# Jitter is common with mediapipe and causes false form alerts.
SMOOTH_N = 6
back_buf  = deque(maxlen=SMOOTH_N)
knee_buf  = deque(maxlen=SMOOTH_N)
torso_buf = deque(maxlen=SMOOTH_N)

# Hip-shoot detection: compare how fast the hip rises vs the shoulder.
# Conventional deadlift coaching cue: "chest and hips rise together."
# If hips accelerate away from the bar first, the lift converts into a
# stiff-leg with the lower back taking the load.
HIP_HIST    = deque(maxlen=12)
SHOULDER_HIST = deque(maxlen=12)

# Rep counting via knee angle.
# Conventional deadlift: knees go from ~80-100 degrees at setup to ~170 at lockout.
rep_count = 0
rep_stage = None   # "down" once knee bends past threshold, "up" at lockout


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        PL = mp_pose.PoseLandmark

        # Pick whichever side has the higher combined visibility score so the
        # checker degrades gracefully when one side is occluded.
        r_vis = lm[PL.RIGHT_SHOULDER].visibility + lm[PL.RIGHT_HIP].visibility
        l_vis = lm[PL.LEFT_SHOULDER].visibility  + lm[PL.LEFT_HIP].visibility

        if r_vis >= l_vis:
            shoulder = to_px(lm[PL.RIGHT_SHOULDER], w, h)
            hip      = to_px(lm[PL.RIGHT_HIP],      w, h)
            knee     = to_px(lm[PL.RIGHT_KNEE],     w, h)
            ankle    = to_px(lm[PL.RIGHT_ANKLE],    w, h)
            ear      = to_px(lm[PL.RIGHT_EAR],      w, h)
        else:
            shoulder = to_px(lm[PL.LEFT_SHOULDER], w, h)
            hip      = to_px(lm[PL.LEFT_HIP],      w, h)
            knee     = to_px(lm[PL.LEFT_KNEE],     w, h)
            ankle    = to_px(lm[PL.LEFT_ANKLE],    w, h)
            ear      = to_px(lm[PL.LEFT_EAR],      w, h)

        # Synthetic mid-spine point: midpoint of shoulder and hip.
        # Mediapipe has no explicit spine landmarks, so this gives a visual
        # reference for where the mid-back sits and anchors the spine curve check.
        mid_spine = ((shoulder[0] + hip[0]) // 2, (shoulder[1] + hip[1]) // 2)

        # ------------------------------------------------------------------ #
        # Angle calculations
        # ------------------------------------------------------------------ #

        # Spine angle: interior angle at the shoulder between ear and hip.
        # When the back is straight, the ear-shoulder vector aligns with the
        # overall torso direction (shoulder->hip), keeping this angle near 180.
        # When the upper back rounds (thoracic kyphosis), the shoulders slump
        # forward and the ear shifts forward relative to the torso line, pulling
        # the angle down. This is far more reliable than shoulder-hip-knee, which
        # drops naturally just from hinging forward with a perfectly flat back.
        back_buf.append(calculate_angle(ear, shoulder, hip))
        back_angle = sum(back_buf) / len(back_buf)

        # Knee angle: interior angle at the knee.
        # Conventional setup: 80-110 degrees. Lockout: ~170 degrees.
        # Persistently below 70 = squatting the weight instead of hinging.
        knee_buf.append(calculate_angle(hip, knee, ankle))
        knee_angle = sum(knee_buf) / len(knee_buf)

        # Torso lean: angle of the torso vector (hip->shoulder) from vertical.
        # 0 = perfectly upright. Expected to be ~45-60 at setup and near 0 at
        # lockout. Tracking this over the rep reveals whether the lifter is
        # achieving full extension at the top.
        dx = shoulder[0] - hip[0]
        dy = shoulder[1] - hip[1]  # positive = downward in image coords
        torso_buf.append(math.degrees(math.atan2(abs(dx), abs(dy))))
        torso_angle = sum(torso_buf) / len(torso_buf)

        # ------------------------------------------------------------------ #
        # Rep counting
        # ------------------------------------------------------------------ #
        hip_y_norm = hip[1] / h
        HIP_HIST.append(hip_y_norm)
        SHOULDER_HIST.append(shoulder[1] / h)

        # Use knee angle transitions to count reps reliably.
        if knee_angle < 115:
            rep_stage = "down"
        elif knee_angle > 155 and rep_stage == "down":
            rep_count += 1
            rep_stage = "up"

        # ------------------------------------------------------------------ #
        # Form checks (ordered highest -> lowest priority)
        # ------------------------------------------------------------------ #
        issues = []

        # 1. Back rounding (thoracic kyphosis).
        # A neutral spine keeps the ear roughly collinear with shoulder and hip,
        # holding the ear-shoulder-hip angle near 180 regardless of how far
        # forward the torso is inclined. When the upper back rounds, the shoulder
        # protracts and the angle drops. Flagging below 150 catches meaningful
        # rounding while still giving a ~30 degree buffer from perfect alignment.
        if back_angle < 150:
            issues.append(("Back rounding - brace and retract your shoulder blades", (0, 0, 255)))

        # 2. Hip shoot.
        # If the hips rise at more than 1.6x the rate of the shoulders over the
        # last 12 frames, the lifter is pushing with the legs and leaving the bar
        # behind, shifting load from the glutes/hamstrings to the lower back.
        if len(HIP_HIST) >= 8 and len(SHOULDER_HIST) >= 8:
            hip_rise      = HIP_HIST[0]      - HIP_HIST[-1]       # positive = rising
            shoulder_rise = SHOULDER_HIST[0] - SHOULDER_HIST[-1]
            if hip_rise > 0.025 and shoulder_rise > 0 and hip_rise > shoulder_rise * 1.6:
                issues.append(("Hip shooting up - drive chest and hips together", (0, 140, 255)))

        # 3. Squatting the weight.
        # Knee angle staying below 70 throughout indicates the lifter is
        # initiating with a vertical shin rather than a hip hinge, turning the
        # deadlift into a squat. This repositions the load incorrectly and
        # usually means the bar is drifting forward.
        if knee_angle < 70:
            issues.append(("Squatting the weight - hinge at the hip more", (0, 165, 255)))

        # 4. Incomplete lockout.
        # At the top of each rep the hips should be fully extended and the
        # torso vertical. Leaning back beyond vertical (hyperextension) is
        # also problematic but cannot be reliably detected from a side view alone.
        if rep_stage == "up" and torso_angle > 30:
            issues.append(("Incomplete lockout - stand tall at the top", (0, 215, 255)))

        # Show the highest-priority issue, or confirm good form.
        if issues:
            form_text, text_color = issues[0]
        else:
            form_text, text_color = "Good form", (0, 200, 0)

        # ------------------------------------------------------------------ #
        # Rendering
        # ------------------------------------------------------------------ #

        # Draw the full mediapipe skeleton first.
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(180, 180, 180), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(80, 180, 255), thickness=2)
        )

        # Highlight the four key joints in the feedback color.
        for pt in [shoulder, hip, knee, ankle]:
            cv2.circle(frame, pt, 8, text_color, -1)

        # Draw the synthetic mid-spine point so the user can visually track
        # whether their mid-back bows outward relative to the shoulder-hip line.
        cv2.circle(frame, mid_spine, 6, (255, 200, 0), -1)

        # Draw spine reference lines: ear -> shoulder -> mid_spine -> hip.
        # A straight spine keeps all four points close to a single line.
        cv2.line(frame, ear,       shoulder,  (255, 200, 0), 2)
        cv2.line(frame, shoulder,  mid_spine, (255, 200, 0), 2)
        cv2.line(frame, mid_spine, hip,       (255, 200, 0), 2)

        # Draw angle values next to the shoulder (spine angle) and knee joints.
        cv2.putText(frame, str(int(back_angle)),
                    (shoulder[0] + 12, shoulder[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, str(int(knee_angle)),
                    (knee[0] + 12, knee[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Semi-transparent HUD panel at the top.
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Spine angle (ear-shldr-hip): {int(back_angle):>3}",
                    (15, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, f"Knee angle       : {int(knee_angle):>3}",
                    (15, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, f"Torso from vert  : {int(torso_angle):>3}",
                    (15, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, f"Reps: {rep_count}",
                    (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 255), 2)

        # Form feedback at the bottom of the frame.
        cv2.putText(frame, form_text,
                    (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

    cv2.imshow("Deadlift Form Checker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
