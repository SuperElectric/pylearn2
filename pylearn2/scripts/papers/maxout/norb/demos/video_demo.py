#! /usr/bin/env python
"""
Video demo. Press 'q' to quit.

Code copied over from OpenCV's tutorial at:
http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video
"""

import cv2

def main():

    cap = cv2.VideoCapture(0)

    keep_going = True

    while(keep_going):
        # Capture frame-by-frame
        keep_going, frame = cap.read()

        if not keep_going:
            print "Error in reading a frame. Exiting..."
        else:
            # Our operations on the frame come here
            # mkg: do we know that the frame is BGR? how can we find out?
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Displays the resulting frame
            cv2.imshow('frame', gray_image)

            # Checks whether user quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                keep_going = False

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
