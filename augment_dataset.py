import cv2
import albumentations as A
import numpy as np
import os
import random
from tqdm import tqdm
from argparse import ArgumentParser
from copy_paste_transform import CopyPaste # Import your custom class
import supervision as sv
from pathlib import Path

# --- Argument Parsing ---
parser = ArgumentParser(description="Augment a YOLO segmentation dataset with advanced controls.")
parser.add_argument('source_dir', type=str, help="Base source directory of the YOLO dataset")
parser.add_argument('-i', '--image-dir', dest="image_dir", type=str, default="train/images", help="Image directory relative to source dir")
parser.add_argument('-l', '--label-dir', dest="label_dir", type=str, default="train/labels", help="Label directory relative to source dir")
parser.add_argument('-o', '--output', dest="output_dir", type=str, help="Output directory for augmented data", default="output")
parser.add_argument('-c', '--class', dest='cls', type=int, default=-1, help="Target class ID for augmentation. If -1, all classes are used.")
parser.add_argument('-n', '--n-images', dest="n_images", type=int, default=0, help="Number of unique images to select for augmentation.")
parser.add_argument('-p', '--permutations', dest="permutations", type=int, default=0, help="Number of augmented versions to generate per original image.")
parser.add_argument('--copy-paste', dest="copy_paste_prob", type=float, default=0.6, help="Probability for copy/paste augmentation to be used.")

args = parser.parse_args()

# --- Configuration ---
BASE_DIR = args.source_dir
SOURCE_IMAGES_DIR = os.path.join(BASE_DIR, args.image_dir)
SOURCE_LABELS_DIR = os.path.join(BASE_DIR, args.label_dir)
OUTPUT_DIR = args.output_dir
TARGET_CLASS_ID = args.cls

# --- Create output directories ---
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

pre_transform = A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=20, p=0.5)])
copypaste_transform = CopyPaste(p=args.copy_paste_prob, pct_objects_paste=1)
post_transform = A.Compose([A.RandomBrightnessContrast(p=0.4, brightness_by_max=False, ensure_safe_range=True)])

print("Loading dataset with supervision...")
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=SOURCE_IMAGES_DIR,
    annotations_directory_path=SOURCE_LABELS_DIR,
    force_masks=True, # Ensure masks are loaded,
    data_yaml_path=os.path.join(BASE_DIR, 'data.yaml')
)

# Create a map for easy lookup
image_keys = list(dataset.annotations.keys())
images_with_target_class = []
if args.cls != -1:
    print(f"Filtering for images containing class ID {TARGET_CLASS_ID}...")
    for i, key in enumerate(image_keys):
        if np.any(dataset.annotations[key].class_id == TARGET_CLASS_ID):
            images_with_target_class.append(i)
    images_to_process_indices = images_with_target_class
else:
    images_to_process_indices = list(range(len(dataset)))

source_pool_indices = images_with_target_class if args.cls != -1 else list(range(len(dataset)))

# --- Logic for n_images and permutations ---
if args.n_images > 0:
    images_to_process_indices = random.sample(images_to_process_indices, min(args.n_images, len(images_to_process_indices)))
IMAGES_TO_GENERATE_PER_ORIGINAL = args.permutations if args.permutations > 0 else (1 if args.n_images > 0 else 0)
if IMAGES_TO_GENERATE_PER_ORIGINAL == 0:
    print("Error: You must specify either --n-images or --permutations.")
    exit()

# --- 3. Main Augmentation Loop ---
print(f"Processing {len(images_to_process_indices)} images, creating {IMAGES_TO_GENERATE_PER_ORIGINAL} versions each...")

for target_idx in tqdm(images_to_process_indices):
    _, target_image, target_detections = dataset[target_idx]
    #breakpoint()
    
    for i in range(IMAGES_TO_GENERATE_PER_ORIGINAL):
        # Stage 1: Pre-CopyPaste Augmentations
        transformed = pre_transform(image=target_image, masks=np.array(target_detections.mask).astype(np.uint8))
        current_image = transformed['image']
        current_masks = transformed['masks']
        # Note: We assume pre_transform doesn't remove masks for simplicity in label tracking
        current_class_labels = list(np.array(target_detections.class_id))

        # Stage 2: Apply CopyPaste
        source_idx = random.choice(source_pool_indices)
        _, source_image, source_detections = dataset[source_idx]
        
        # Format the bboxes as the custom class expects: (*coords, mask_index)
        source_bboxes_formatted = [list(xyxy) + [idx] for idx, xyxy in enumerate(source_detections.xyxy)]
        
        pasted = copypaste_transform(
            image=current_image,
            masks=current_masks,
            paste_image=source_image,
            paste_masks=np.array(source_detections.mask).astype(np.uint8),
            paste_bboxes=source_bboxes_formatted
        )
        current_image = pasted['image']
        pasted_masks = pasted['masks']
        
        # Reconstruct class labels
        num_pasted = len(pasted_masks) - len(current_masks)
        final_class_labels = current_class_labels + list(np.array(source_detections.class_id)[-num_pasted:])
        
        # Stage 3: Post-CopyPaste Augmentations
        final_augmented = post_transform(image=current_image, masks=pasted_masks)
        final_image = final_augmented['image']
        final_masks = final_augmented['masks']

        # Save results
        original_basename = Path(image_keys[target_idx]).stem
        new_filename_base = f"{original_basename}_aug_{i}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, "images", f"{new_filename_base}.jpg"), final_image)
        
        h, w, _ = final_image.shape
        with open(os.path.join(OUTPUT_DIR, "labels", f"{new_filename_base}.txt"), 'w') as f:
            for class_id, mask in zip(final_class_labels, final_masks):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea).squeeze()
                    if contour.ndim == 1 or len(contour) < 3: continue
                    normalized = contour / np.array([w, h])
                    f.write(f"{class_id} " + " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in normalized]) + "\n")

print("Integrated augmentation complete!")
