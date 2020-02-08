import imutils
import cv2

vs = cv2.VideoCapture('PATH_TO_VIDEO')

firstFrame = None
MIN = 700

while True:
    frame = vs.read()[1]

    # the end of the video
    if frame is None:
        break

    frame = imutils.resize(frame, width=1080, height=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (19, 19), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < MIN:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the frame and record if the user presses a key
    cv2.imshow("result", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
