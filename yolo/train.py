from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x-seg.pt")  # load a pretrained model (recommended for training)
model.to("cuda:0")

# Train the model
model.train(
    data="coco8-seg.yaml",
    epochs=100,
    imgsz=1024,
    batch=2,
    project="yolo-seg",
    name="yolov8x-seg",
    plots=True,
)
