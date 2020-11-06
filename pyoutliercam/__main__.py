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
parser.add_argument("--width", help="Width of frame to process", type=int, const=1, nargs='?', default=680)
parser.add_argument("--height", help="Height of frame to process", type=int, const=1, nargs='?', default=420)
parser.add_argument("--cell-width", help="Width of cell", type=int, const=1, nargs='?', default=170)
parser.add_argument("--cell-height", help="Height of cell", type=int, const=1, nargs='?', default=105)
parser.add_argument("--debug", help="Debug mode on or off", type=bool, const=1, nargs='?', default=False)
parser.add_argument("--image", help="Image file to process (Required if mode is capture-objects)")
parser.add_argument("--object-proposal-algo", choices=["selective-search", "edge-boxes"], help="Object proposal algo to use (Required if mode is capture-objects)", const=1, nargs='?', default="selective-search")
parser.add_argument("--output-dir", help="Output directory for object proposals (Required if mode is capture-objects)", const=1, nargs='?', default="./")
parser.add_argument("--object-width", help="Width of the desired object to evaluate", type=int, const=1, nargs='?', default=250)
parser.add_argument("--object-height", help="Height of the desired object to evaluate", type=int, const=1, nargs='?', default=250)

args = parser.parse_args()

if __name__ == '__main__':
    mode    = args.mode
    debug   = args.debug

    if mode == "alarm":
        cam         = args.cam
        width       = args.width
        height      = args.height
        cell_width  = args.cell_width
        cell_height = args.cell_height

        cmd = Alarm(cam=cam, width=width, height=height, cell_width=cell_width, cell_height=cell_height, debug=debug)

        print("Capture at camera: %d" % (cmd.cam))
        print("Frame dimensions for processing %d x %d" % (cmd.width, cmd.height))
        
        while(True):

            cmd.run()
            cmd.display()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cmd.clean()
    elif mode == "capture-objects":
        print("Not yet implemented...")
#        image                   = args.image
#        object_proposal_algo    = args.object_proposal_algo
#        output_dir              = args.output_dir
#        object_width            = args.object_width
#        object_height           = args.object_height
#
#        cmd = CaptureObjects(image, object_proposal_algo, output_dir, object_width, object_height)
#        cmd.execute()
#
#        if debug:
#            cmd.display()


    print("Done.")
