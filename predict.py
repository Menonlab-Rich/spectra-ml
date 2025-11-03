import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from argparse import ArgumentParser

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
mask_annotator = sv.MaskAnnotator(opacity=0.4)
box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()


# --- 4. Process Images in Batches ---
print(f"Found {len(image_paths)} images. Processing in batches of {args.batch_size}...")

for i in range(0, len(image_paths), args.batch_size):
    batch_paths = image_paths[i:i + args.batch_size]
    
    # --- Perform Inference ---
    # `save_txt=True` handles the requirement to save YOLO format labels
    # The results are saved to a 'predict' folder by default
    results = model.predict(
        source=batch_paths,
        conf=args.confidence,
        save_txt=True,
        project=args.dest,
        name=f'batch_{i // args.batch_size}'
    )

    # --- Create Mosaic for the current batch ---
    annotated_frames = []
    for result, img_path in zip(results, batch_paths):
        # Read the original image
        frame = cv2.imread(str(img_path))

        # Convert ultralytics results to supervision's Detections object
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter detections by confidence if needed (predict already does this)
        # detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]

        # Generate labels for each detection
        labels = [
            f"{model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate the frame with masks, boxes, and labels
        annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        annotated_frames.append(annotated_frame)

    # Plot the grid of annotated frames for the current batch
    if annotated_frames:
        mosaic_image = sv.plot_images_grid(
            images=annotated_frames,
            grid_size=(2, 2), # Adjust grid size based on BATCH_SIZE
            titles=[os.path.basename(p) for p in batch_paths],
            
        )

        if mosaic_image is not None:
        
        # Save the mosaic image
            mosaic_filename = f"mosaic_batch_{i // args.batch_size}.jpg"
            mosaic_filepath = os.path.join(args.dest, mosaic_filename)
            cv2.imwrite(mosaic_filepath, mosaic_image)
            print(f"Saved mosaic to {mosaic_filepath}")

print("Processing complete.")
