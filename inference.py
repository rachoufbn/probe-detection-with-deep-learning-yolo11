import os
from ultralytics import YOLO
from lib import get_input_folder_path, get_base_dir, CONFIG

base_dir = get_base_dir()
input_image_folder_path = get_input_folder_path()

final_model_file_name = "final_model" + CONFIG["model_export"]["extension"]
final_model_path = os.path.join(base_dir, final_model_file_name)

# Batch set to 1 so that images are processed one by one (according to the specification), performance could be improved by using larger batches
batch_size = 1

model = YOLO(final_model_path)
results = model.predict(
    batch=batch_size,
    source=input_image_folder_path,
    device=CONFIG["train"]["device"],
    imgsz=CONFIG["image_size"],
    conf=CONFIG["inference"]["confidence_threshold"],
    save=True
)