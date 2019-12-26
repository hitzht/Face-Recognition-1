import face_model
import argparse
import cv2
import sys
import numpy as np
import os
import csv
import socket
import pickle
import struct ## new
import zlib

HOST = ''
PORT = 8001

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')

s.listen(1)
print('Socket now listening')

def parse_arguments():

    parser = argparse.ArgumentParser(description='face model test')

    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--data-img', default='./Images', help='path to data images')
    parser.add_argument('--skip-frame', default=100, help='number of frames to skip')
    parser.add_argument('--model', default='../models/model-r100-ii/model/model,0', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.5, type=float, help='ver dist threshold') # 1.24

    return parser.parse_args()


def face_recognition(frame, model, data, name):
    # frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
        
    ret = model.get_input(frame) 

    id = []

    if ret is None: # if there are no faces
        print("Cannot find any faces")
        cv2.imshow("IMG", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return   
        return

    F2, bboxs = ret 

    for i in range(len(F2)):
        f2 = model.get_feature(F2[i])  
        
        check = 0 # To check if no face match will return unknown
                    
        for j in range(len(data)):
            if check == 1:
                break
            f1 = data[j]

            sim = np.dot(f1, f2.T)

            if sim > 0.5:
                check = 1

                color = (0,255,255)
                font = cv2.FONT_HERSHEY_SIMPLEX

                bbox = bboxs[i, 0:4]
                bbox = bbox.astype(np.int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, name[j], (bbox[0], bbox[1]), font, 0.4, (0,255,255), 1, cv2.LINE_AA)

                print(name[j])
                id.append(name[j])

                continue

        if check == 0:
            color = (0,0,255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            bbox = bboxs[i, 0:4]
            bbox = bbox.astype(np.int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, "Unknown", (bbox[0], bbox[1]), font, 0.4, (255,255,255), 1, cv2.LINE_AA)

            print("unknown")

    cv2.imshow("IMG", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return 1
    
    return id
    # video_writer.write(frame)


def main(readerd, readern):
    args = parse_arguments()
    skip_num = 0

    model = face_model.FaceModel(args)

    name = []
    for row in readern:
        name.append(row[0])
    
    embedded = []
    for row in readerd:
        embedded.append(np.asarray(row, dtype=np.float32))

    conn, addr = s.accept()
    data = b""
    payload_size = struct.calcsize(">L")

    while True:
        if skip_num % args.skip_frame != 0:
            skip_num += 1
            continue
            
        while len(data) < payload_size:
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
       
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is not None:
            ret = face_recognition(frame, model, embedded, name)
            if ret == 1:
                break
            print("RET = ", ret)
            data_string = pickle.dumps(ret)
            conn.send(data_string)

        skip_num += 1 

    # When everything is done, release the capture
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # open file
    with open('Data/data.csv', 'rt') as d:
        with open('Data/name.csv', 'rt') as n:
            readerd = csv.reader(d)
            readern = csv.reader(n)
            main(readerd, readern)
           
            d.close()
            n.close()
