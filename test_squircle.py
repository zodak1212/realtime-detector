import cv2
import numpy as np

img = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    cv2.putText(img, "test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    print("Success text")
except Exception as e:
    print(e)
