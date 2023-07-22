import cv2 as cv
from aruco_detector import Aruco_Detector
import os


def main():
    
    data_folder = os.path.join(os.path.dirname(__file__), "calibration_data")

    video_cap = cv.VideoCapture(0)
    detector = Aruco_Detector("logitech_webcam_data.npz", data_folder)

    while (True):

        # Read frame
        success, frame = video_cap.read()

        # Detect tags and draw on frame
        tags, frame = detector.detect_tags(frame)
        

        # Show image in a window
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    video_cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()