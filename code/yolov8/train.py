# @dong
# v8 train 230703

from ultralytics import YOLO

# train det
model = YOLO("workspace/weights/yolov8s-det.pt")

model.train(data='ultralytics/datasets/VOC.yaml', epochs=30, imgsz=640, batch=256)