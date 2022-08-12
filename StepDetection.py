import math
import sys
from functools import partial

import mediapipe as mp
import time

import numpy as np
from cv2 import norm
from PIL import Image, ImageTk
import cv2
import imutils
import MainPage

from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLabel,QComboBox
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QMutex, QWaitCondition
import cgitb
cgitb.enable(format = 'text')

def list_ports():
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing.
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

class handDetector():
    static_image_mode = False,
    max_num_hands = 2,
    model_complexity = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
    def __init__(self, mode=False,complexity=1,smooth=True,segmentation=False,smoothSegmentation=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.segmentation = segmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.cx, self.cy, self.cx1, self.cy1 = 0, 0, 0, 0

        self.mpFoots = mp.solutions.pose
        self.foots = self.mpFoots.Pose(self.mode,self.complexity,self.smooth,self.segmentation,self.smoothSegmentation,self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findFoot(self, img, draw=True):
        global count,normalfootspandistance,firstdistanceCMofCam,step,beforestepcm,stepstart,stepend,maxfootspandistance,minfootspandistance,firstdegree
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.foots.process(imgRGB)
        if self.results.pose_landmarks:
            self.cx, self.cy, self.cx1, self.cy1 = 0, 0, 0, 0
            if draw:
                POSE_CONNECTIONS = frozenset([(27, 29),(29, 31),(27, 31), (28, 30),
                                              (30, 32), (32, 28), (31, 32)])
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, POSE_CONNECTIONS)
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    if id == 31:
                        self.cx, self.cy = int(lm.x * w), int(lm.y * h)
                        a = np.array([lm.x,lm.y])

                    if id == 30:
                        self.cx, self.cy = int(lm.x * w), int(lm.y * h)
                    if id == 32:
                        self.cx1, self.cy1 = int(lm.x * w), int(lm.y * h)
                        b = np.array([lm.x,lm.y])
                    if id == 12:
                        self.cx2, self.cy2 = int(lm.x * w), int(lm.y * h)
                    if id == 11:
                        self.cx3, self.cy3 = int(lm.x * w), int(lm.y * h)

                if self.cx!=0 and self.cy!=0 and self.cx1!=0 and self.cy1!=0 and self.cx2!=0 and self.cy2!=0 and self.cx3!=0 and self.cy3!=0 :
                    cheekbonedistance = norm((self.cx2,self.cy2),(self.cx3,self.cy3))
                    footspandistance = norm((self.cx,self.cy),(self.cx1,self.cy1))

                    A1,B1,C1=coff1
                    A,B,C=coff
                    distanceCM=A*cheekbonedistance**2+B*cheekbonedistance+C
                    distanceScreenCm =A1*int(distanceCM)**2+B1*int(distanceCM)+C1
                    footspandistanceCM=footspandistance*distanceCM/distanceScreenCm
                    theta = math.acos((a[1] - b[1]) * (-b[1]) / (math.sqrt(
                        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) * b[1]))
                    degree = int(180 / math.pi) * theta


                    print("angle:"+str(degree))

                    if count==0:
                        normalfootspandistance=footspandistanceCM
                        firstdistanceCMofCam=distanceCM
                        firstdegree=degree
                        count+=1

                    if abs(degree)>(abs(firstdegree)+10) :
                        stepend=False
                        stepstart = True
                    if stepstart==True and abs(degree)<(abs(firstdegree)+10) :
                        stepend =True
                    if stepstart ==True and stepend==True :
                        step+=1
                        stepstart=False
                        stepend=False
                    print("normalangle: "+str(firstdegree))
                    print("foot: " + str(footspandistanceCM))
                    print("distance: " + str(distanceCM))
                    print("normal: "+str(normalfootspandistance))
                    print("first: "+str(firstdistanceCMofCam))
                    print("step: "+str(step))
                    cv2.putText(img, "Angle : "+str(int(degree)), (0,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1)
                    cv2.putText(img, "Foot Span Distance : "+str(int(footspandistanceCM)), (0,90), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1)
                    cv2.putText(img, "Distance : "+str(int(distanceCM)), (0,130), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1)
                    cv2.putText(img, "Step : "+str(step), (0,170), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        #if self.results.multi_hand_landmarks:
            #myHand = self.results.multi_hand_landmarks[handNo]
        #     for id, lm in enumerate(myHand.landmark):
        #         # print(id, lm)
        #         h, w, c = img.shape
        #         cx, cy = int(lm.x * w), int(lm.y * h)
        #         # print(id, cx, cy)
        #         lmList.append([id, cx, cy])
        #         if draw:
        #             cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        # return lmList
video = cv2.VideoCapture(0)
cTime=0
pTime=0
class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, mutex, condition):
        super().__init__()
        self.mutex = mutex
        self.condition = condition

    def run(self):
        global video,cTime,pTime
        while running:
            try:
                success, img = video.read()
                if success == True:
                    img = imutils.resize(img, width=750)
                    img = detector.findFoot(img)

                    # lmlist=detector.findPosition(img)
                    # if(len(lmlist)!=0):
                    #     print(lmlist[4])
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime

                    cv2.putText(img, str(int(fps)), (700, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(img.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
                    p = convertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                    self.changePixmap.emit(p)
                    self.condition.wait(self.mutex)
            except Exception as e:
                print(e)

class FootTrackingPage(QMainWindow):
    def __init__(self,items,cTime,pTime):
        super().__init__()
        try:
            self.mutex = QMutex()
            self.condition = QWaitCondition()
            self.setWindowTitle("Foot Tracking")
            # Create a Qt widget, which will be our window.


            self.setFixedSize(QSize(1280, 720))
            self.qr = self.frameGeometry()
            self.cp=self.screen().availableGeometry().center()
            self.qr.moveCenter(self.cp)
            self.move(self.qr.topLeft())
            self.lbl_video = QLabel(self)
            self.lbl_video.move(10,10)
            self.lbl_video.setFixedSize(640,480)

            self.com=QComboBox(self)
            self.com.move(1100,125)
            self.com.setFixedSize(60,30)
            self.com.addItems([str(x) for x in items])
            # Set the central widget of the Window.
            self.com.currentIndexChanged.connect(self.getselectedindex)
            global detector
            detector = handDetector()
            self.button = QPushButton("Kamerayı aç", self)
            self.button.resize(90, 32)
            self.button.move(800, 125)
            self.button.clicked.connect(partial(self.ac,cTime,pTime))

            self.button1 = QPushButton("Kamerayı Kapat", self)
            self.button1.setFixedSize(90, 32)
            self.button1.move(920,125)
            self.button1.clicked.connect(self.kapa)
        except Exception as e:
            print(e)

    def setImage(self, image):
        self.mutex.lock()
        try:
            self.lbl_video.setPixmap(QPixmap.fromImage(image))
        except Exception as e:
            print(e)
        finally:
            self.mutex.unlock()
            self.condition.wakeAll()

    def getselectedindex(self):
        global video
        video = cv2.VideoCapture(self.com.currentIndex())
        print(self.com.currentIndex())
        print(video)


    # def showcam( lbl_video, cTime, pTime):
    #     global running, video
    #
    #     success, img = video.read()
    #
    #     if success == True and running == True:
    #         img = imutils.resize(img, width=750)
    #         img = detector.findFoot(img)
    #
    #         # lmlist=detector.findPosition(img)
    #         # if(len(lmlist)!=0):
    #         #     print(lmlist[4])
    #         cTime = time.time()
    #         fps = 1 / (cTime - pTime)
    #         pTime = cTime
    #
    #         cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         img = Image.fromarray(img)

    def ac(self, cTime, pTime):
        global running, count, items,th
        running = True
        count = 0
        time.sleep(4)
        video = cv2.VideoCapture(self.com.currentIndex())
        th = Thread(mutex = self.mutex,condition=self.condition)
        th.changePixmap.connect(self.setImage)
        th.start()

    def kapa(self):
        global running,th
        running = False
        th.exit(0)
        self.lbl_video.clear()

    def closeEvent(self, event):
        print("asdasd")
        self.close()


def main():
    global video,coff,coff1,normalfootspandistance,count,firstdistanceCMofCam,step,beforestepcm,stepstart,stepend,maxfootspandistance,minfootspandistance,items,firstdegree
    step=0
    maxfootspandistance = 0
    minfootspandistance = 0
    firstdegree=0
    stepstart=False
    stepend=False
    items = []
    objectwidth = [615, 530, 430, 385, 340, 315, 280, 245, 225, 205, 190]
    realdistancecam = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    y = [745, 803, 781, 816, 824, 859, 848, 816, 818, 807, 806]
    x = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    coff = np.polyfit(objectwidth, realdistancecam, 2)  # y=Ax^2 + Bx + C
    coff1 = np.polyfit(x, y, 2) # y=Ax^9 + Bx^8 + Cx^7+ Dx^6 + Ex^5 + F
    print("asdsa")


def openmainpage():
    pTime = 0
    cTime = 0
    available_ports, working_ports, non_working_ports = list_ports()
    items.extend(working_ports)
    app = QApplication(sys.argv)
    foottrackingpage = FootTrackingPage(items, cTime, pTime)
    foottrackingpage.show()
    app.exec()

if __name__ == "__main__":
    main()
    openmainpage()
