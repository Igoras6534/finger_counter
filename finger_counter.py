import cv2
import time 
import os
import hand_tracking_modoule as htm



cam = cv2.VideoCapture(1)
pTime=0

wCam,hCam=640,480
cam.set(3,wCam)
cam.set(4,hCam)

folder_path="fingers"
myList=os.listdir(folder_path)
overlayList=[]


for impath in myList:
    image=cv2.imread(f"{folder_path}/{impath}")
    overlayList.append(image)

tipsIDS=[4,8,12,16,20]
detector=htm.handDetector(maxHands=1,detectionCon=0.7)

while True:
    _,frame=cam.read()
    frame=cv2.flip(frame,1)
    detector.findHands(frame,draw=True)
    lmlist=detector.findPosition(frame,draw=False)

    if len(lmlist)!=0:
        fingers=[]
        if lmlist[tipsIDS[0]][1]>lmlist[tipsIDS[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if lmlist[tipsIDS[id]][2]<lmlist[tipsIDS[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers=fingers.count(1)
        print(fingers)
        h,w,c=overlayList[0].shape
        frame[0:h,0:w] = overlayList[total_fingers]


    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime


    cv2.putText(frame,f"FPS:{fps}",(30,400),cv2.FONT_HERSHEY_PLAIN
                ,3,(255,0,0),2)
    cv2.imshow("Cam",frame)

    if cv2.waitKey(1)==ord("q"):
        break



