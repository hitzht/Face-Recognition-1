import sys
sys.path.append('/opt/anaconda3/envs/InsightFace/lib/python3.7/site-packages')

import face_model
import argparse
import cv2
import numpy as np
import os
import csv
import vptree
import math
import time

def parse_arguments():

    parser = argparse.ArgumentParser(description='face model test')

    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--data-img', default='./Images', help='path to data images')
    parser.add_argument('--model', default='../models/model-r100-ii/model/model,0', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.5, type=float, help='ver dist threshold') # 1.24

    return parser.parse_args()

def euclidean(p1, p2):
    v1 = p1[1]
    v2 = p2[1]

    return math.acos(np.dot(v1, v2.T))
    # return -(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def main(readerd, readern):
    args = parse_arguments()

    model = face_model.FaceModel(args)

 
    name = []
    for row in readern:
        name.append(row)

    data = []
    for i, row in enumerate(readerd):
        temp = [i, np.asarray(row, dtype=np.float32)]
        data.append(temp)

    # Form VP tree
    tree = vptree.VPTree(data, euclidean)

    capture = cv2.VideoCapture(0)

    while(capture.isOpened()):
        ret, frame = capture.read()

        if ret is None:
            print("Cannot open the video!")
            break
        
        if frame is None:
            continue

        frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
        
        ret = model.get_input(frame) 

        if ret is None: # if there are no faces
            print("Cannot find any faces")
            cv2.imshow("IMG", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   
            continue

        F2, bboxs = ret 

        start_time = time.time()
        for i in range(len(F2)):
            f2 = model.get_feature(F2[i]) 

            result = tree.get_nearest_neighbor([0, f2])

            sim = np.dot(f2, result[1][1].T)
            print(name[result[1][0]][0])
            if sim > 0.5:
                    color = (0,255,255)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    bbox = bboxs[i, 0:4]
                    bbox = bbox.astype(np.int)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, name[result[1][0]][0], (bbox[0], bbox[1]), font, 0.4, (0,255,255), 1, cv2.LINE_AA)
            else:
                color = (0,0,255)
                font = cv2.FONT_HERSHEY_SIMPLEX

                bbox = bboxs[i, 0:4]
                bbox = bbox.astype(np.int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, "Unknown", (bbox[0], bbox[1]), font, 0.4, (255,255,255), 1, cv2.LINE_AA)

                print("unknown")
                        
        cv2.imshow("IMG", frame)
        end_time = time.time()

        print("fps = ", 1 / (end_time - start_time))
        # video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    capture.release()
    cv2.destroyAllWindows()

    # dist = np.sum(np.square(f1-f2))
    # print("DIST = ", dist)

    # sim = np.dot(f1, f2.T)
    # print("SIM = ", sim) # Cosine similarity - la gia tri cos(theta) cua hai vector -1 <= x <=1 qqq

if __name__ == '__main__':

    # open file
    with open('Data/data.csv', 'r') as d:
        with open('Data/name.csv', 'r') as n:
            readerd = csv.reader(d)
            readern = csv.reader(n)
            main(readerd, readern)
           
            d.close()
            n.close()

