import os
from ultralytics import YOLO
from lib import get_input_folder_path, get_base_dir, CONFIG

base_dir = get_base_dir()
input_image_folder_path = get_input_folder_path()

final_model_file_name = "final_model" + CONFIG["model_export"]["extension"]
final_model_path = os.path.join(base_dir, final_model_file_name)

model = YOLO(final_model_path)
results = model.predict(
    source=input_image_folder_path,
    imgsz=CONFIG["image_size"],
    save=True
)