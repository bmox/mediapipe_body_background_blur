import cv2
import mediapipe as mp
import numpy as np
import datetime
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
pose=mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
def find_human(image,draw=True):
    BG_COLOR = (0, 255, 0)
    image.flags.writeable = False
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    body_parts=[]
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    body_parts.append([id,cx,cy]) 
    else:
        body_parts.append("None")
    image.flags.writeable = True
    try:
        annotated_image = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
    except:
        annotated_image = image.copy()
    if draw:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    return annotated_image, body_parts      

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
out = cv2.VideoWriter(
    f"{current_time}.mp4", fourcc, 30, frame_size)

while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    image,body_parts =find_human(image,draw=False)
    # out.write(image)
    cv2.imshow('MediaPipe Pose', image )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()
