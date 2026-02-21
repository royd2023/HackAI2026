import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

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
    """Calculate angle between three points (in degrees)"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) -
        math.atan2(a[1]-b[1], a[0]-b[0])
    )
    return abs(ang)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # --- Get key points (right side) ---
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hip      = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        knee     = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle    = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Convert to pixel coordinates
        shoulder_xy = (int(shoulder.x * w), int(shoulder.y * h))
        hip_xy      = (int(hip.x * w), int(hip.y * h))
        knee_xy     = (int(knee.x * w), int(knee.y * h))
        ankle_xy    = (int(ankle.x * w), int(ankle.y * h))

        # --- Calculate angles ---
        back_angle = calculate_angle(shoulder_xy, hip_xy, knee_xy)
        knee_angle = calculate_angle(hip_xy, knee_xy, ankle_xy)

        # --- Deadlift Form Heuristics ---
        form_text = "Good Form"
        color = (0, 255, 0)

        # Back rounding check
        if back_angle < 150:
            form_text = "Back Rounding!"
            color = (0, 0, 255)

        # Squatting too much
        elif knee_angle < 100:
            form_text = "Too Much Knee Bend"
            color = (0, 165, 255)

        # Display feedback
        cv2.putText(frame, f"Back Angle: {int(back_angle)}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"Knee Angle: {int(knee_angle)}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, form_text,
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Draw skeleton
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Deadlift Form Checker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()