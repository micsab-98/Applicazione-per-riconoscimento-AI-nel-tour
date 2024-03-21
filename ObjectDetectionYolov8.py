from PIL import Image
from ultralytics import YOLO

# Load a model
model = YOLO('Yolov8/best_50epochs_datasetClean.pt')  # load a custom model

# Predict with the model
results = model('immaginePanoramica.jpg')  # predict on an image

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #im.show()  # show image
    im.save('resultYolov8.jpg')  # save image

# Extract bounding boxes, classes, names, and confidences
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()

# Iterate through the results
for box, cls, conf in zip(boxes, classes, confidences):
    x1, y1, x2, y2 = box
    confidence = conf
    detected_class = cls
    name = names[int(cls)]
    print(x1)