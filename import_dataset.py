import os
from lib import YoloLabelConversion, get_input_folder_path, get_base_dir, CONFIG

train_val_test_weights = tuple(CONFIG["import_dataset"]["train_val_test_weights"])

base_dir = get_base_dir()
import_dataset_dir = get_input_folder_path()

image_folder_path = os.path.join(import_dataset_dir, "probe_images")
labels_json_path = os.path.join(import_dataset_dir, "probe_labels.json")
dataset_folder_path = os.path.join(base_dir, "dataset")

if not os.path.isdir(image_folder_path):
    raise Exception("Probe dataset image folder not found.")

if not os.path.isfile(labels_json_path):
    raise Exception("Labels json file not found.")

yolo_label_conversion = YoloLabelConversion(
    image_folder_path,
    labels_json_path,
    dataset_folder_path,
    train_val_test_weights
)

yolo_label_conversion.convert()