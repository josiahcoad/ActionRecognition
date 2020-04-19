import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RBG)
    frames = frame.reshape(1, *frame.shape)
    points = frames2points(frames)
    x = preprocess(points)[0]
    prob = singlepred(model, x)

    cv2.imshow('frame', bgra)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out = cv2.imwrite('capture.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()