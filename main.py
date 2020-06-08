import cv2
import onnxruntime as rt
from imageprocessing import process

cap = cv2.VideoCapture(0)
sess = rt.InferenceSession("facemaskdetect.onnx")

while(True):
    ret, frame = cap.read()
    im = process(frame,sess)
    cv2.imshow("result",im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()