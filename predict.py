import os
from argparse import ArgumentParser

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

parser = ArgumentParser(description="Perform inferences using a custom YOLO model.")
parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Path to the YOLO model weights.")
parser.add_argument("--src", type=str, required=True, help="Path to the images to use for predictions.")
parser.add_argument("--dest", type=str, default="results/predict", help="Path to store the results to.")
parser.add_argument("--batch-size", type=int, dest="batch_size", default=4, help="Size of batches to perform predictions on.")
parser.add_argument("--confidence", type=float, help="Confidence threshold to consider a positive detection")

args = parser.parse_args()

# --- 1. Configuration ---

# Create results directory if it doesn't exist
os.makedirs(args.dest, exist_ok=True)


# --- 2. Load Model and Get Image Paths ---
model = YOLO(args.model)
image_paths = sv.list_files_with_extensions(
    directory=args.src,
    extensions=["jpg", "jpeg", "png"]
)


# --- 3. Setup Annotators from Supervision ---
# These helpers will draw the masks, boxes, and labels
mask_annotator = sv.MaskAnnotator(opacity=0.3)
# This was being called in your original script but wasn't defined
label_annotator = sv.LabelAnnotator()


# --- 4. Process Images in Batches ---
print(f"Found {len(image_paths)} images. Processing in batches of {args.batch_size}...")

for i in range(0, len(image_paths), args.batch_size):
    batch_paths = image_paths[i:i + args.batch_size]
    
    # --- Perform Inference ---
    # We removed save_txt, project, and name
    # We will handle all saving manually
    results = model.predict(
        source=batch_paths,
        conf=args.confidence,
    )

    # --- Process and Save Individual Results ---
    for result, img_path in zip(results, batch_paths):
        
        # --- A. Define File Paths ---
        # Get the original filename without the extension
        # e.g., /path/to/image01.jpg -> image01
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # 1. index_raw.jpg
        raw_save_path = os.path.join(args.dest, f"{base_filename}_raw.jpg")
        
        # 2. index_labeled.jpg
        labeled_save_path = os.path.join(args.dest, f"{base_filename}_labeled.jpg")
        
        # 3. index.txt
        txt_save_path = os.path.join(args.dest, f"{base_filename}.txt")

        
        # --- B. Read and Save Raw Image ---
        frame = cv2.imread(str(img_path))
        cv2.imwrite(raw_save_path, frame)

        # --- C. Save YOLO TXT Output ---
        # Use the result object's method to save to our custom path
        result.save_txt(txt_save_path)

        # --- D. Annotate and Save Labeled Image ---
        
        # Convert ultralytics results to supervision's Detections object
        detections = sv.Detections.from_ultralytics(result)
        
        # Generate labels for each detection
        labels = [
            f"{model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate the frame with masks, boxes, and labels
        annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Save the final annotated image
        cv2.imwrite(labeled_save_path, annotated_frame)
        
        print(f"Saved results for {os.path.basename(img_path)} as {base_filename}.*")


print("Processing complete.")
