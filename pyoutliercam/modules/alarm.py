import cv2
import numpy as np

class Alarm:
    def __init__(self, cam=0, width=680, height=420, cell_width=170, cell_height=105, debug=True):
        self.cam    = cam
        self.width  = width
        self.height = height
        self.cap    = cv2.VideoCapture(cam)
        self.debug  = debug

        self.cell_width     = cell_width
        self.cell_height    = cell_height

        self.label_main_window  = "Main"
        self.label_debug_frame  = "Debug"

        self.color      = (0, 0, 255)
        self.thickness  = 2

    def run(self):
        self.ret, self.frame    = self.cap.read()
        self.bw_frame           = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.processed_frame    = cv2.resize(self.frame, (self.width, self.height))
        self.debug_frame        = self.processed_frame.copy()

        # Define areas
        x = 0
        y = 0

        self.areas  = []

        while y < self.height:
            while x < self.width:
                start_point = (x, y)
                end_point   = (x + self.cell_width, y + self.cell_height)

                _x = x + self.cell_width
                _y = y + self.cell_height

                roi = self.processed_frame[x:_x, y:_y]

                self.areas.append({ "roi": roi, "start_point": start_point, "end_point": end_point })

                x += self.cell_width

            y += self.cell_height
            x = 0

    def display(self):
        cv2.imshow(self.label_main_window, self.frame)

        if self.debug:
            for area in self.areas:
                self.debug_frame = cv2.rectangle(self.debug_frame, area["start_point"], area["end_point"], self.color, self.thickness)

            cv2.imshow(self.label_debug_frame, self.debug_frame)

    def clean(self):
        self.cap.release()
        cv2.destroyAllWindows()
