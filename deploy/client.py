import cv2
import io
import socket
import struct
import time
import pickle
import zlib
from threading import Thread
import time

HOST = ''
# HOST = ''
PORT = 8001

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('34.80.169.132', 8000))
client_socket.connect((HOST, PORT))
connection = client_socket.makefile('wb')

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.01)
        
        self.capture.release()
        cv2.destroyAllWindows()
        exit(1)

    def show_frame(self):
        # Stream frames in main program
        result, frame = cv2.imencode('.jpg', self.frame, encode_param)
        data = zlib.compress(pickle.dumps(frame, 0))
        data = pickle.dumps(frame, 0)
        size = len(data)
        client_socket.sendall(struct.pack(">L", size) + data)

if __name__ == '__main__':
    # video_stream_widget = VideoStreamWidget("rtsp://admin:dol@2018@192.168.1.64/1")
    video_stream_widget = VideoStreamWidget(0)

    while True:
        try:
            video_stream_widget.show_frame()

            data = client_socket.recv(4096)
            data_arr = pickle.loads(data)
            print(data_arr)
        except AttributeError:
            pass

