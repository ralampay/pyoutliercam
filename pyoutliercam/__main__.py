import sys
import argparse
import os
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from collections import deque

from modules.alarm import Alarm

parser = argparse.ArgumentParser(description="PyOutlierCam: Anomaly Detector for Cameras")
parser.add_argument("--mode", choices=["capture-objects", "alarm"], help="Mode to run in", required=True)
parser.add_argument("--cam", help="Camera index to open (Required if mode is alarm)", type=int, const=1, nargs='?', default=0)
parser.add_argument("--width", help="Width of frame to process", type=int, const=1, nargs='?', default=600)
parser.add_argument("--height", help="Height of frame to process", type=int, const=1, nargs='?', default=400)
parser.add_argument("--debug", help="Debug mode on or off", type=bool, const=1, nargs='?', default=False)

args = parser.parse_args()

if __name__ == '__main__':
    mode    = args.mode
    width   = args.width
    height  = args.height
    debug   = args.debug

    if mode == "capture-objects":
        pass
    elif mode == "alarm":
        cam = args.cam
        cmd = Alarm(cam, width, height, debug)

        print("Capture at camera: %d" % (cmd.cam))
        print("Frame dimensions for processing %d x %d" % (cmd.width, cmd.height))
        
        while(True):

            cmd.run()
            cmd.display()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cmd.clean()

    print("Done.")
