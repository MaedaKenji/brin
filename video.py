import cv2
from ultralytics import YOLO

model = YOLO(r"D:\Code\Python\brin\best.pt")

results = model(r"D:\Code\Python\brin\lab.mp4", stream=True)

for r in results:
    frame = r.plot()
    # Resize the frame to 720p (1280x720) for better visualization
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("YOLO Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()