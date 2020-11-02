import sys
import argparse
import os
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

parser = argparse.ArgumentParser(description="PyOutlierCam: Anomaly Detector for Cameras")
parser.add_argument("--cam", help="Camera index to open", required=True, type=int)
parser.add_argument("--width", help="Width of frame to process", type=int, const=1, nargs='?', default=600)
parser.add_argument("--height", help="Height of frame to process", type=int, const=1, nargs='?', default=400)
parser.add_argument("--debug", help="Debug mode on or off", type=bool, const=1, nargs='?', default=False)

args = parser.parse_args()

if __name__ == '__main__':
    cam = args.cam
    cap = cv2.VideoCapture(cam)

    width   = args.width
    height  = args.height
    debug   = args.debug

    print("Capture at camera: %d" % (cam))
    print("Frame dimensions for processing %d x %d" % (width, height))

    while(True):
        ret, frame = cap.read()

        # Display windows
        cv2.imshow("Main", frame)

        if debug:
            processed_frame = cv2.resize(frame, (width, height))

            cv2.imshow("Processed", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Done.")
