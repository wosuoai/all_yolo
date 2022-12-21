import numpy as np
import cv2 as cv
from multiprocessing import Process
import time
import json
from collections import deque
from threading import Thread
import pymysql
from fastapi import FastAPI
import uvicorn
from yolo_api import predict
import cv2


def videoSaveAp(pts_ong,shapes):
    Tim = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    read_video = cv.VideoWriter(Tim + ".mp4", fourcc, 30, (shapes[1],shapes[0]))
    for frame in list(pts_ong):
        read_video.write(frame)
        cv.waitKey(10)
    read_video.release()

def Sequy(text):
    print("=================《{}》=======================".format(str(text)))

minute = 2
second = 10

anomaly_f = False
scope_num = 2
A_fps_minute_scope = minute * 60 * 30
A_fps_second_scope = second * 30
print("{} minute = {}".format(minute,A_fps_minute_scope))

@app.post('/video/interval')
def process():
    global p

    all_list = []
    for i in p:
        a_list = "rtsp://xx:xxx@xxx:{prot}//Streaming/Channels/2".format(prot=i)
        all_list.append(a_list)

    processes = []
    for ab in all_list:
        t = Process(target=video, args=(ab,))
        processes.append(t)

    for i in range(len(all_list)):
        processes[i].start()

    for j in range(len(all_list)):
        processes[j].join()

def video(video_info):
    global anomaly_f
    pts = deque(maxlen=A_fps_minute_scope)

    cap = cv.VideoCapture(video_info)
    print("N : ", cap.isOpened())
    s = 0
    count = 0
    while True:
        success, frame = cap.read()

        if not success:
            break
        s += 1
        shapes = frame.shape
        pts.append(frame)
        # --------------------------------
        detections = predict(img_path=frame,model_path="yolov5s.pt")
        for ids in detections:
            cv2.rectangle(img=frame, pt1=(int(ids[0][0]), int(ids[0][1])), pt2=(int(ids[0][2]), int(ids[0][3])),
                          color=(0, 255, 0), thickness=2, lineType=2)
            text = "{} - {}".format(ids[1], ids[2])
            cv2.putText(frame, text, org=(int(ids[0][0]) + 10, int(ids[0][1]) + 10), fontFace=1, fontScale=1,
                        color=(0, 0, 255), thickness=2, lineType=2)

            if ids[1] == "person":
                anomaly_f = True

        # -----------------------------------
        if anomaly_f or count >= 1:
            anomaly_f = False
            Sequy("触发异常点")
            count += 1
        print("count:", count)

        if A_fps_minute_scope / scope_num == count and count >= 1:
            count = 0
            opone = list(pts)
            Sequy("视频截取")
            start = time.time()
            t = Thread(target=videoSaveAp, args=(opone, shapes))
            t.start()
            t.join()
            end = time.time()
            print("time:", (end - start))

        cv.imshow("img", frame)

        if cv.waitKey(27) in [ord("q"), 27]:
            break


if __name__ == '__main__':
    p = input("请输入端口号：").split()
    process()