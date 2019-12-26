
import sys
sys.path.append('/opt/anaconda3/envs/InsightFace/lib/python3.7/site-packages')
import csv
import face_model
import argparse
import cv2
import numpy as np
import os

def parse_arguments():

    parser = argparse.ArgumentParser(description='face model test')

    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--data-img', default='./Images', help='path to data images')
    parser.add_argument('--skip-frame', default=20, help='number of frames to skip')
    parser.add_argument('--model', default='../models/model-r100-ii/model/model,0', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

    return parser.parse_args()

def main():
    args = parse_arguments()

    model = face_model.FaceModel(args)

    path = args.data_img
    name = [x for x in os.listdir(path) if x != '.DS_Store']
    files = [os.path.join(path, f) for f in name]

    with open('Data/data.csv', 'w') as csvData:
        with open('Data/name.csv', 'w') as csvName:

            writerData = csv.writer(csvData)
            writerName = csv.writer(csvName)

            for i in range(len(files)):
                file = files[i]
                img_names = os.listdir(file)
                img_paths = [os.path.join(file, f) for f in img_names]

                for img_path in img_paths:
                    print(img_path)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    try:
                        F1, _ = model.get_input(img)
                    except:
                        continue
                    f1 = model.get_feature(F1[0])

                    writerData.writerow(f1)
                    writerName.writerow([name[i]])
        
        csvData.close()   
        csvName.close()         

if __name__ == '__main__':
    main()

