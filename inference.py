# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # pretrained YOLO26n model

# Run batched inference on a list of images
# results = model([r"C:\Users\Agus\Downloads\right.jpg"], stream=True)  # return a generator of Results objects
# results = model([r"C:\Users\Agus\Downloads\front.jpg"], stream=True) 
results = model([r"C:\Users\Agus\Downloads\sitting.jpg"], stream=True) 



# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk