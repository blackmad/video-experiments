import sys
import getopt
import cv2
import numpy as np

cap = cv2.VideoCapture(sys.argv[1])

subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
print(f'input {width}x{height}')

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(width),int(height)))

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'num frames {length}')

framesProcessed = 0
while True:
    ret, frame = cap.read()
    if not ret:
      break
    mask = subtractor.apply(frame)
    frame2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    out.write(frame2)
    framesProcessed += 1
    if framesProcessed % 100 == 0:
      print(f'processsed {framesProcessed} of {length} {framesProcessed*100/length:.2f}%')

out.release()
cap.release()
cv2.destroyAllWindows()
