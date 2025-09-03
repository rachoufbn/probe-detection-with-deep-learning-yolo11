import os
from yolo_label_conversion import YoloLabelConversion 
from ultralytics import YOLO

base_dir = os.path.dirname(
    os.path.abspath(__file__)
)

image_folder_path = os.path.join(base_dir, "probe_dataset", "probe_images")
labels_json_path = os.path.join(base_dir, "probe_dataset", "probe_labels.json")

dataset_path = os.path.join(base_dir, "dataset")

yolo_label_conversion = YoloLabelConversion(
    image_folder_path,
    labels_json_path,
    dataset_path
)

yolo_label_conversion.convert()

# path to your dataset yaml (created earlier with train/val paths)
data_yaml = os.path.join(dataset_path, "data.yaml")

# 1) load a pretrained backbone
# start small (yolo11n.pt), scale up if you have GPU room (s, m, l, x)
model = YOLO("yolo11n.pt")

# 2) train
model.train(
    data=data_yaml,      # dataset yaml
    epochs=100,          # start with 100, extend if needed
    #imgsz=640,           # standard input size
    batch=16,            # adjust to fit GPU memory
    device='cpu'             # GPU index, or 'cpu' if no GPU
)

# 3) validate on the val split
metrics = model.val(data=data_yaml)
print(metrics)

