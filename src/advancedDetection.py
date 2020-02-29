import cv2
from imutils.video import VideoStream
from detector import MotionDetector

if __name__ == "__main__":

    # cap = cv2.VideoCapture("PATH_TO_VIDEO")
    cap = VideoStream(src=0).start()

    detector = MotionDetector(bg_history=20,
                              group_boxes=True,
                              expansion_step=5,
                              pixel_compression_ratio=0.4,
                              min_area=500)

    while True:
        # Capture frame-by-frame
        frame = cap.read()

        if frame is None:
            break
        boxes = detector.detect(frame)

        scale = detector.pixel_compression_ratio
        for b in boxes:
            cv2.rectangle(frame, (int(b[0] / scale), int(b[1] / scale)),
                          (int(b[0] / scale) + int(b[2] / scale), int(b[1] / scale) + int(b[3] / scale)),
                          (0, 255, 0), 1)

        cv2.imshow('last_frame', frame)
        cv2.imshow('diff_frame', detector.color_movement)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
