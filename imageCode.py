import sys
import time

import cv2

import tensorflow as tf
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
import os
from PIL import Image
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import uic, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication
from PyQt5.uic import loadUi
from PyQt5.uic.properties import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


import numpy as np
import tensorflow as tf
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
import os
from detect_video_simple import appConfig
Ui_Form = uic.loadUiType('main.ui')[0]

class AppConfig(QMainWindow, QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)
        self.initUI()

    def initUI(self):
        # 창 설정
        self.setWindowTitle('Mango')
        # self.setWindowIcon(QIcon('vaccination.png'))
        # self.resize(700, 500)
        #self.center()
        self.startCam.clicked.connect(lambda :self.StartCamAndResult())
        #self.show()


    def StartCamAndResult(self):
        self.StartCam()
        self.second = ResultClass()
        self.second.exec()
        self.show()

    def StartCam(self):
        MODEL_PATH = './checkpoints/yolov4-416'
        IOU_THRESHOLD = 0.45
        SCORE_THRESHOLD = 0.25
        INPUT_SIZE = 416
        # load model
        saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']



        cap = cv2.VideoCapture(0)
        str = "Mango Classification"
        captured_num = 0
        count = 0

        while cap.isOpened():
            ret, img = cap.read()

            cv2.putText(img, str, (100, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))

            if not ret:
                break

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            img_input = img_input / 255.
            img_input = img_input[np.newaxis, ...].astype(np.float32)
            img_input = tf.constant(img_input)

            pred_bbox = infer(img_input)

            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=IOU_THRESHOLD,
                score_threshold=SCORE_THRESHOLD
            )

            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

            for i in range(valid_detections.numpy()[0]):
                if int(classes.numpy()[0][i]) < 0 or int(classes.numpy()[0][i]) > 2: continue

                self.class_ind = int(classes.numpy()[0][i])

                if self.class_ind == 0 or self.class_ind == 1:
                    print(count, "카운트")
                    count += 1

            result = utils.draw_bbox(img, pred_bbox)

            result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

            cv2.imshow('MangoClassfication', result)
            if cv2.waitKey(1) == ord('q'):
                break
                # read frame from webcam

            if count < 5: continue

            if cap.isOpened():

                status, frame = cap.read()

                if not status:
                    break

                    # display output
                path = 'C:/Users/zkdht/PycharmProjects/yolo_v4/images'
                file_list = os.listdir(path)
                print(len(file_list), file_list)


                captured_num = captured_num + 1
                cv2.imwrite('C:/Users/zkdht/PycharmProjects/yolo_v4/images/1.jpg', frame)

                dst = cv2.resize(frame, dsize=(1080, 720), interpolation=cv2.INTER_AREA)
                #cv2.imshow("captured frames", dst)
                count = 0
                cap.release()
                time.sleep(1)
                cv2.destroyAllWindows()
        #lasc=lastClass(self)
        #lasc.fruit(ResultClass.Fruit_percent)
    #def detect_video(self):



class ResultClass(QDialog):
    def __init__(self):
        super(ResultClass, self).__init__()
        loadUi("dialog.ui", self)
        self.initUI()
        self.show()
        #self.show()

    def initUI(self):
        #배경제거된 망고
        ripped1, ripped_days1 = self.printcolor()

        bg_img = 'images/a.jpg'
        pixmap1 = QPixmap(bg_img)
        pixmap1 = pixmap1.scaled(731, 480)
        self.bgOff.setPixmap(pixmap1)

        #컬러플롯
        img = 'images/color_plot.png'
        im = Image.open(img)
        im = im.crop((75, 200, 570, 290))  # left, upper, right, lower

        im.save(img, dpi=(1500, 1500))

        pixmap = QPixmap(img)
        """pixmap = QPixmap(imgPath) #"""
        pixmap = pixmap.scaled(731, 111)#, QtCore.Qt.KeepAspectRatio
        self.plot_image.setPixmap(pixmap)

        #그라데이션
        gradient_img = 'images/gradient.png'
        pixmap2 = QPixmap(gradient_img)
        self.color_graph.setPixmap(pixmap2)





        #슬라이더 표시

        #self.verticalSlider

        self.verticalSlider.setValue(self.Fruit_percent)
        self.sl = QSlider(QtCore.Qt.Vertical, self)
        self.sl.setStyleSheet('background-color: rgba(255, 255, 255, 0);')
        self.sl.setMinimum(0)
        self.sl.setMaximum(12)
        self.sl.setValue(ripped_days1)
        self.sl.setTickPosition(QSlider.TicksLeft)
        self.sl.setTickInterval(1)
        self.sl.setSingleStep(2)


        self.sl.valueChanged.connect(lambda value: print(value))

        for index, value in enumerate(range(12, 0 - 1, -1)):
            if index%2 ==0 and value%2 == 0:
                label = QLabel("{}".format(value))
                label.setStyleSheet('font: 75 20pt "카페24 써라운드";background-color: rgba(255, 255, 255, 0);')
                self.gridLayout.addWidget(label, index, 0, QtCore.Qt.AlignRight)


        self.gridLayout.addWidget(self.sl, 0, 1, 12 - 0 + 1, 1, QtCore.Qt.AlignRight)

        if ripped_days1 > 0:
            self.need_day.setText("[섭씨 23~25도]에서 약 "+ str(ripped_days1)+ "일 만큼 추가로 숙성 필요")

        """def initUI(self):
            # 창 설정
            self.setWindowTitle('Mango')
            # self.setWindowIcon(QIcon('vaccination.png'))
            # self.resize(700, 500)
            # self.center()
            self.show()"""


        self.last_btn.clicked.connect(self.show_about_dialog)
        self.goback_btn.clicked.connect(self.exit)

    def exit(self):
        self.close()


    def centroid_histogram(self, clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist

    def plot_colors(self, hist, centroids):
        if self.histsum < 1:
            a = 300
        elif self.histsum > 1:
            a = 350

        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, a, 3), dtype="uint8")
        startX = 0

        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar

    def printcolor(self):

        # Read image
        img = cv2.imread('images/1.jpg')
        hh, ww = img.shape[:2]

        # threshold on white
        # Define lower and uppper limits
        lower = np.array([110, 110, 110])
        upper = np.array([255, 255, 255])

        # Create mask to only select black
        thresh = cv2.inRange(img, lower, upper)

        # apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # invert morp image
        mask = 255 - morph

        # apply mask to image
        result = cv2.bitwise_and(img, img, mask=mask)

        # save results
        cv2.imwrite('pills_thresh.jpg', thresh)
        cv2.imwrite('pills_morph.jpg', morph)
        cv2.imwrite('pills_mask.jpg', mask)
        cv2.imwrite('images/a.jpg', result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))  # 3차원이 2차원
        k = 6  # 예제는 5개로 나누겠습니다
        self.histsum = 0
        self.clt = KMeans(n_clusters=k)
        self.clt.fit(image)
        for center in self.clt.cluster_centers_:
            print(center)

        hist = self.centroid_histogram(self.clt)

        for i in range(5):

            if np.any(self.clt.cluster_centers_[i] < 1):
                hist = np.delete(hist, i)
                self.clt.cluster_centers_ = np.delete(self.clt.cluster_centers_, i, 0)
        for i in range(5):
            hist[i] = hist[i] / (hist[0] + hist[1] + hist[2] + hist[3] + hist[4])
        for i in range(5):
            hist[i] = hist[i] * 1.8
        for i in range(5):
            self.histsum += hist[i]



        bar = self.plot_colors(hist, self.clt.cluster_centers_)

        # show our color bart
        plt.figure()
        plt.axis("off")

        plt.imshow(bar)
        #plt.show()
        plt.savefig('images/color_plot.png')


        sum = 0
        black = 0
        self.overripp = 0



        for i in range(5):

            pixel = np.uint8([[self.clt.cluster_centers_[i]]])  # 색상 정보를 하나의 픽셀로 변환한다
            hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)

            hsv = hsv[0][0]  # 색상 정보만 가져온다

            print("RGB: ", self.clt.cluster_centers_[i])
            print("HSV: ", hsv)

            if hsv[0] < 40:
                sum = sum + hist[i] / (hist[0] + hist[1] + hist[2] + hist[3] + hist[4]) * 100
                if hsv[2] < 100:
                    self.overripp = self.overripp + hist[i] / (hist[0] + hist[1] + hist[2] + hist[3]+hist[4]) * 100

        self.Fruit_percent = int(sum)

        ripped = "숙성도 :"+ str(self.Fruit_percent)+ "%" #"과잉숙성"+ str(int(overripp)) + "%"

        ripped_days = int(12 - 12 * self.Fruit_percent / 100)
        #overripp2 = "섭씨 30℃ 에서 약"+ str(ripped_days)+ "일 만큼 추가로 숙성시킨다."


        return ripped, ripped_days

    def show_about_dialog(self):
        qmsgBox = QMessageBox(self)
        qmsgBox.setBaseSize(QSize(1080,720))
        qmsgBox.setMinimumHeight(1080)

        a = ""
        if self.Fruit_percent > 30:
            if self.overripp < 30:
                text = """
                                       <center>
                                           <h1>MANGO</h1><br/>
                                           <img src=images/box.png width=200 height=200>
                                           <h2> 지금당장 판매해야합니다 </h2>
                                       </center>
                                       <p>Version 0.0.1</p>
                                       """
                qmsgBox.about(self, "Mango", text)
                qmsgBox.setStyleSheet('.QMessageBox {color: rgb(0, 0, 0)}QLabel{background:transparent;color:#fff}')
                # pixmap = QPixmap(imgPath)

            if self.overripp >= 40:
                # 쓰레기통
                text = """
                           <center>
                               <h1 >MANGO</h1><br/>
                            <img src=images/garbage_can.png width=300 height=300>
                                <p style="font-size: 30px;font-weight:bold"> 폐기해야합니다.</p>
                           </center>
                           <p>Version 0.0.1</p>
                        """
                qmsgBox.about(self, "Mango", text)
                qmsgBox.setStyleSheet('.QMessageBox {color: rgb(0, 0, 0)}QLabel{background:transparent;color:#fff}')
                box = 'images/garbage_can.png'
                a = "폐기해야합니다"

        elif self.Fruit_percent < 30:
            # 타이머
            text = """
                      <center>
            <h1>MANGO</h1><br/>
                <img src=images/timer.png width=200 height=200><h2> 조금 기다렸다가 포장해야합니다.</h2>
                      </center>
                      <p>Version 0.0.1</p>
                                  """
            qmsgBox.about(self, "Mango", text)
            qmsgBox.setStyleSheet('.QMessageBox {color: rgb(0, 0, 0)}QLabel{background:transparent;color:#fff}')

        else:
            text = """
                                 <center>
                       <h1>MANGO</h1><br/>
                           <img src=images/timer.png width=200 height=200>"다시실행"<p> 
                                 </center>
                                 <p>Version 0.0.1</p>
                                             """
            qmsgBox.about(self, "Mango", text)
            qmsgBox.setStyleSheet('.QMessageBox {color: rgb(0, 0, 0)}QLabel{background:transparent;color:#fff}')


        """boxim = Image.open(box)
        pixmap3 = QPixmap(boxim)
        pixmap3 = pixmap3.scaled(870, 501, QtCore.Qt.KeepAspectRatio)
        self.plot_image.setPixmap(pixmap3)
        self.notice_label.setText(a)"""

        #self.show()





if __name__ == '__main__':
    app = QApplication(sys.argv)
    #app.setStyleSheet(qss)
    """ app_icon = QtGui.QIcon()
    app_icon.addFile('vaccination.png', QtCore.QSize(16, 16))
    app_icon.addFile('vaccination.png', QtCore.QSize(24, 24))
    app_icon.addFile('vaccination.png', QtCore.QSize(32, 32))
    app_icon.addFile('vaccination.png', QtCore.QSize(48, 48))
    app_icon.addFile('vaccination.png', QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)"""
    widget = QtWidgets.QStackedWidget()

    # 레이아웃 인스턴스 생성
    mainWindow = AppConfig()


    mainWindow.show()
    app.exec_()



