import os, sys, shutil
from lib import get_base_dir, CONFIG
from ultralytics import YOLO

base_dir = get_base_dir()
training_runs_dir = os.path.join(base_dir, "runs", "detect")

available_models = {}

# Search training runs for available models
with os.scandir(training_runs_dir) as d:
    for training_run_dir in d:
        if (
            training_run_dir.is_dir() and
            os.path.isdir(os.path.join(training_run_dir.path, "weights")) and
            os.path.isfile(os.path.join(training_run_dir.path, "weights", "best.pt"))
        ):
            model_name = training_run_dir.name
            best_model_path = os.path.join(training_run_dir.path, "weights", "best.pt")
            available_models[model_name] = best_model_path

print("Available trained models:")
print(" ".join(available_models.keys()))

# Prompt user to select a model
while True:
    selected_model_name = input("Enter model name to export (or 'exit' to quit): ")
    if selected_model_name.lower() == 'exit':
        sys.exit(0)
    if selected_model_name in available_models:
        break
    print("Invalid model name. Please try again.")

selected_model_path = available_models[selected_model_name]
print(f"Exporting model: {selected_model_name} from {selected_model_path}")

# Export to ONNX format
model = YOLO(selected_model_path)
model.export(format=CONFIG["model_export"]["format"], imgsz=tuple(CONFIG["image_size"]))

# Move exported model to base directory with standard name
old_exported_model_path = os.path.splitext(selected_model_path)[0] + CONFIG["model_export"]["extension"]
new_exported_model_path = os.path.join(base_dir, "final_model" + CONFIG["model_export"]["extension"])
shutil.copy2(old_exported_model_path, new_exported_model_path)