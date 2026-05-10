import cv2
from ultralytics import YOLO

model = YOLO(r"D:\Code\Python\brin\best.pt")

input_video_path = r"D:\Code\Python\brin\lab.mp4"
output_video_path = r"D:\Code\Python\brin\lab_out.mp4"

# Get original video FPS to use for the output video
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps == 0:
    fps = 30.0
cap.release()

# Define the codec and create VideoWriter object
# Using 'mp4v' codec for MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (1280, 720))

results = model(input_video_path, stream=True)

for r in results:
    frame = r.plot()
    # Resize the frame to 720p (1280x720) for better visualization
    frame = cv2.resize(frame, (1280, 720))
    
    # Write the frame into the output video
    out.write(frame)

# Release the VideoWriter
out.release()
print(f"Video saved to {output_video_path}")
