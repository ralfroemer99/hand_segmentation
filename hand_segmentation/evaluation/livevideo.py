from segmenthands import *

cap = cv2.VideoCapture(0)
i = 0
while (True):
    ret, frame = cap.read()

    frame = cv2.resize(frame, (384, 288))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if i % 20 == 0:
        i = 0
        mask = SegmentHands(rgb)
        colmask = getcoloredMask(frame, mask)

    cv2.imshow('color', np.hstack((frame, colmask)))
    key = cv2.waitKey(24)
    if key & 0xFF == ord('q'):
        break
    i += 1
cap.release()
cv2.destroyAllWindows()

