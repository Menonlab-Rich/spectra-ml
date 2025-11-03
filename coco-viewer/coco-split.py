import supervision as sv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image_dir", help="Path to your images")
parser.add_argument("annotations_path", help="Path to your annotations file")
parser.add_argument("output_dir", help="Path to your output dataset")

args = parser.parse_args()

# Paths to your COCO dataset

# Load the COCO dataset
dataset = sv.DetectionDataset.from_coco(
    images_directory_path=args.image_dir,
    annotations_path=args.annotations_path
)

# Split the dataset into 80% for training and 20% for testing
train_dataset, test_dataset = dataset.split(split_ratio=0.8)

# Save the split datasets in YOLO format (which is convenient for YOLO training)
train_dataset.as_yolo(
    images_directory_path=os.path.join(args.output_dir, "train/images"),
    annotations_directory_path=os.path.join(args.output_dir, "train/labels"),
    data_yaml_path=os.path.join(args.output_dir, "data.yaml"),
    yolo_format="polygon"
)

test_dataset.as_yolo(
    images_directory_path=os.path.join(args.output_dir, "test/images"),
    annotations_directory_path=os.path.join(args.output_dir, "test/labels"),
    data_yaml_path=os.path.join(args.output_dir, "data.yaml"), 
    yolo_format="polygon"
)

print(f"Dataset split and saved in YOLO format at: {args.output_dir}")
