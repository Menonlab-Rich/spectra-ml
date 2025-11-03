import supervision as sv
import yaml
from pathlib import Path
import os
import argparse

# --- 1. Set up Argument Parser ---
parser = argparse.ArgumentParser(description="Split a YOLO dataset into training and testing sets.")
parser.add_argument('--source-images', type=str, default="../metavision/yolo_dataset/images",
                    help="Path to the source directory containing all images.")
parser.add_argument('--source-labels', type=str, default="../metavision/yolo_dataset/labels",
                    help="Path to the source directory containing all YOLO label files.")
parser.add_argument('--config', type=str, default="./conf/model/yolov8_config.yaml",
                    help="Path to your dataset configuration YAML file.")
parser.add_argument('--base-path', type=str, default=None,
                    help="Optional base path for the output dataset if the path in the config is relative.")
parser.add_argument('--split-ratio', type=float, default=0.8,
                    help="The ratio for the training set split (e.g., 0.8 for an 80/20 split).")

args = parser.parse_args()

# --- Configuration (now from command line) ---
SOURCE_IMAGES_DIR = args.source_images
SOURCE_LABELS_DIR = args.source_labels
CONFIG_YAML_PATH = args.config
BASE_PATH_ARG = args.base_path
SPLIT_RATIO = args.split_ratio

# 2. Load the dataset configuration file ⚙️
print(f"Loading configuration from: {CONFIG_YAML_PATH}")
with open(CONFIG_YAML_PATH, 'r') as f:
    config = yaml.safe_load(f)

# --- Determine the final output base path ---
output_base_path_str = config['path']
if BASE_PATH_ARG and not os.path.isabs(output_base_path_str):
    print(f"Path in config is relative. Joining with provided base path: {BASE_PATH_ARG}")
    base_path = Path(BASE_PATH_ARG) / output_base_path_str
else:
    base_path = Path(output_base_path_str)

# 3. Load the source dataset using supervision 📚
print("Loading source dataset...")
ds = sv.DetectionDataset.from_yolo(
    images_directory_path=SOURCE_IMAGES_DIR,
    annotations_directory_path=SOURCE_LABELS_DIR,
    data_yaml_path=CONFIG_YAML_PATH,
    force_masks=True
)

# 4. Split the dataset into training and testing sets ✂️
train_ds, test_ds = ds.split(split_ratio=SPLIT_RATIO, random_state=42, shuffle=True)

print("-" * 30)
print(f"Dataset split complete.")
print(f"Training set size: {len(train_ds)}")
print(f"Test set size: {len(test_ds)}")
print("-" * 30)

# 5. Construct full output paths from the YAML config 🗺️
train_images_path = base_path / config['train']
train_labels_path = train_images_path.parent / 'labels'
test_images_path = base_path / config['val']
test_labels_path = test_images_path.parent / 'labels'

# 6. Save the split datasets to the configured locations 💾
print(f"Saving training set to: {train_images_path.parent}")
train_ds.as_yolo(
    images_directory_path=train_images_path,
    annotations_directory_path=train_labels_path
)

print(f"Saving test set to: {test_images_path.parent}")
test_ds.as_yolo(
    images_directory_path=test_images_path,
    annotations_directory_path=test_labels_path
)

# Create the final data.yaml in the output directory
with open(CONFIG_YAML_PATH) as stream:
    data_dict = yaml.safe_load(stream)
    data_dict['path'] = str(base_path.resolve())

with open(base_path / 'data.yaml', 'w') as f:
    f.write(yaml.dump(data_dict))

print("\n✅ Splitting and saving complete!")
