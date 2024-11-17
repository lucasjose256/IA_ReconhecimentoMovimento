import cv2
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

image_path = 'imagem3.jpeg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = pose.process(image_rgb)

if result.pose_landmarks:
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        result.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )

    cv2.imshow('Esqueleto', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhum esqueleto detectado na imagem.")

pose.close()
