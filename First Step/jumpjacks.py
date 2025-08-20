import cv2
import mediapipe as mp
import  math
video=cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
Pose = mp_pose.Pose(min_tracking_confidence=0.5,min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils
check=True
while True:
    sucess,img=video.read()
    videoRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img,points,mp_pose.POSE_CONNECTIONS)
    h,w,_=img.shape
    contador=0
    if points:
        peDY=int( points.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y*h)
        peDX=int( points.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x*w)

        peEY=int( points.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y*h)
        peEX=int( points.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x*w)

        maoEY = int(points.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y*h)
        maoEX = int(points.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x*w)

        maoDY = int(points.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y*h)
        maoDX = int(points.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x*w)


        distMAO=math.hypot(maoDX-maoEX,maoDY-maoEY)
        distPE=math.hypot(peDX-peEX,peDY-peEY)

        if check==True and distMAO<=150 and distPE>=150:
            contador=+1
            check=False
        if distMAO >150 and distPE <150:
            check=True

        texto=f'QTD {contador}'
        cv2.putText(img,texto,(40,200),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),5)
    cv2.imshow("PÓ DE CHINELOS",img)



    cv2.waitKey(40)