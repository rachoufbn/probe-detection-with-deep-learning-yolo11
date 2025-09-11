import os
from ultralytics import YOLO
from lib import get_base_dir, CONFIG

base_dir = get_base_dir()

dataset_folder_path = os.path.join(base_dir, "dataset")
data_yaml_path = os.path.join(dataset_folder_path, "data.yaml")

# Load a pretrained backbone
model = YOLO(CONFIG["train"]["starting_model"])

# Train on custom dataset
model.train(
    data=data_yaml_path,
    epochs=CONFIG["train"]["epochs"],
    patience=CONFIG["train"]["patience"],
    imgsz=CONFIG["image_size"],
    batch=CONFIG["train"]["batch_size"],
    cache=CONFIG["train"]["cache"],
    device=CONFIG["train"]["device"],
    workers=CONFIG["train"]["workers"],
    optimizer=CONFIG["train"]["optimizer"],
    degrees=CONFIG["train"]["degrees"],
    shear=CONFIG["train"]["shear"],
    plots=True
)

# Test on the test split
metrics_test = model.val(
    data=data_yaml_path,
    imgsz=CONFIG["image_size"],
    batch=CONFIG["train"]["batch_size"],
    device=CONFIG["train"]["device"],
    workers=CONFIG["train"]["workers"],
    plots=True,
    split='test'
)
print("\nTest metrics: ", metrics_test)