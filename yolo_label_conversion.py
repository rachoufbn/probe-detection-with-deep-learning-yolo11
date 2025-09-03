import os, json, shutil
from ultralytics.data.split import autosplit

class YoloLabelConversion:
    def __init__(
        self,
        image_folder_path,
        labels_json_path,
        dataset_folder_path
    ):
        self.image_folder_path = image_folder_path
        self.labels_json_path = labels_json_path

        self.dataset_folder_path = dataset_folder_path
        self.dataset_images_folder_path = os.path.join(dataset_folder_path, "images")
        self.dataset_labels_folder_path = os.path.join(dataset_folder_path, "labels")

        self.train_val_test_weights = (0.7, 0.2, 0.1)

    def convert(self):
        self._initialise_dataset_dirs()

        with open(self.labels_json_path) as f:
            labels = json.load(f)

        bboxes_by_image_id = {}

        # Key bboxes by image_id
        for annotation in labels["annotations"]:
            image_id = annotation["image_id"]
            bbox = annotation["bbox"]

            if image_id not in bboxes_by_image_id:
                bboxes_by_image_id[image_id] = []

            bboxes_by_image_id[image_id].append(bbox)

        # 
        for image in labels["images"]:
            bboxes = bboxes_by_image_id[image["id"]]
            yolo_label_file_content = ""

            for bbox in bboxes:
                yolo_bbox_string = self._convert_to_yolo_format(image["width"], image["height"], bbox)
                yolo_label_file_content += yolo_bbox_string + "\n"
            
            self._create_yolo_label_file(image["file_name"], yolo_label_file_content)

            image_file_path = os.path.join(self.image_folder_path, image["file_name"])
            shutil.copy2(image_file_path, self.dataset_images_folder_path)

        # Train / Val split
        autosplit(
            path=self.dataset_images_folder_path,
            weights=self.train_val_test_weights,
            annotated_only=True
        )

    def _convert_to_yolo_format(self, image_width, image_height, bbox):
        bbox_x_min, bbox_y_min, bbox_width, bbox_height = bbox
        bbox_x_center = bbox_x_min + (bbox_width/2)
        bbox_y_center = bbox_y_min + (bbox_height/2)

        # Corresponds to class index defined in data.yaml
        class_id = 0
        
        # YOLO coordinates are normalised from 0 to 1
        yolo_bbox_x_center = bbox_x_center/image_width
        yolo_bbox_y_center = bbox_y_center/image_height
        yolo_bbox_width = bbox_width/image_width
        yolo_bbox_height = bbox_height/image_height

        yolo_bbox_string = f"{class_id} {yolo_bbox_x_center} {yolo_bbox_y_center} {yolo_bbox_width} {yolo_bbox_height}"

        return yolo_bbox_string

    def _create_yolo_label_file(self, image_file_name, yolo_label_file_content):
        yolo_label_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        yolo_label_file_path = os.path.join(self.dataset_labels_folder_path, yolo_label_file_name)
        with open(yolo_label_file_path, "w") as f:
            f.write(yolo_label_file_content)

    def _create_dir(self, path):
        os.makedirs(path, exist_ok=True)

    def _initialise_dataset_dirs(self):
        dataset_dirs = [self.dataset_images_folder_path, self.dataset_labels_folder_path]

        for dataset_dir in dataset_dirs:

            # Make sure directory exists
            self._create_dir(dataset_dir)

            # Make sure directory is empty
            for file_name in os.listdir(dataset_dir):
                file_path = os.path.join(dataset_dir, file_name)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
