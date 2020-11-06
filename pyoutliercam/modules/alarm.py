import cv2
import numpy as np

class Alarm:
    def __init__(self, cam=0, width=600, height=400, debug=True):
        self.cam    = cam
        self.width  = width
        self.height = height
        self.cap    = cv2.VideoCapture(cam)
        self.debug  = debug

        self.label_main_window      = "Main"
        self.label_processed_frame  = "Processed"

    def run(self):
        self.ret, self.frame    = self.cap.read()
        self.bw_frame           = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.processed_frame    = cv2.resize(self.bw_frame, (self.width, self.height))

    def display(self):
        cv2.imshow(self.label_main_window, self.frame)

        if self.debug:
            cv2.imshow(self.label_processed_frame, self.processed_frame)

    def clean(self):
        self.cap.release()
        cv2.destroyAllWindows()
