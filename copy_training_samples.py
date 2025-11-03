import os
import shutil
import random
import yaml
from collections import defaultdict
from tqdm import tqdm

# =================================================================================
# --- CONFIG: PLEASE EDIT THESE VALUES ---
# =================================================================================

# Path to the root of your YOLO dataset
DATASET_PATH = "./conf/model/yolov8_config.yaml"

# Which data split to process ('train', 'val', or 'test')
DATA_SPLIT = "train"

# The categories you want to select images from
TARGET_CATEGORIES = ["mkate", "mcherry"]

# How many random images to select for EACH category
NUM_IMAGES_PER_CATEGORY = 3

# Where to save the new, smaller dataset
OUTPUT_DIR = "copied_samples"

# =================================================================================
# --- SCRIPT LOGIC ---
# =================================================================================

def select_yolo_subset():
    """
    Selects and copies a random subset of a YOLO dataset based on specified categories.
    """
    print("--- Starting YOLO Dataset Subset Selection ---")

    # Validate paths
    if not os.path.exists(DATASET_PATH):
        print(f"Error: 'data.yaml' not found at {DATASET_PATH}")
        return
    with open(DATASET_PATH) as stream:
        ds = yaml.safe_load(stream)
    root_path = ds['path']
    split_path = os.path.join(root_path, ds[DATA_SPLIT].split('/')[0])
    image_dir = os.path.join(split_path, 'images')
    label_dir = os.path.join(split_path, 'labels')



    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return
    if not os.path.isdir(label_dir):
        print(f"Error: Label directory not found at {label_dir}")
        return

    # 2. Load YAML and map target categories to class IDs
    print(f"Loading dataset configuration from {DATASET_PATH}...")
    with open(DATASET_PATH, 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    class_names = data_yaml['names']
    try:
        target_class_ids = [class_names.index(name) for name in TARGET_CATEGORIES]
    except ValueError as e:
        print(f"Error: One of the target categories not found in data.yaml: {e}")
        return

    print(f"Targeting categories: {TARGET_CATEGORIES} (IDs: {target_class_ids})")

    # 3. Scan labels to find images containing target categories
    print("Scanning label files to find relevant images...")
    category_to_images = defaultdict(list)
    for label_file in tqdm(os.listdir(label_dir)):
        if not label_file.endswith('.txt'):
            continue
        
        basename = os.path.splitext(label_file)[0]
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                try:
                    class_id = int(line.split()[0])
                    if class_id in target_class_ids:
                        category_to_images[class_id].append(basename)
                except (ValueError, IndexError):
                    continue # Skip malformed lines

    # 4. Select random images for each category
    print("Selecting random images...")
    files_to_copy = set()
    for class_id in target_class_ids:
        category_name = class_names[class_id]
        image_list = list(set(category_to_images.get(class_id, []))) # Use set to get unique images
        
        num_available = len(image_list)
        num_to_select = min(NUM_IMAGES_PER_CATEGORY, num_available)

        if num_available < NUM_IMAGES_PER_CATEGORY:
            print(f"Warning: Found only {num_available} images for '{category_name}'. Selecting all of them.")
        else:
            print(f"Found {num_available} images for '{category_name}'. Randomly selecting {num_to_select}.")

        selected_images = random.sample(image_list, num_to_select)
        files_to_copy.update(selected_images)

    # 5. Create output directories and copy files
    print(f"\nTotal unique images to copy: {len(files_to_copy)}")
    output_image_dir = os.path.join(OUTPUT_DIR, "images", DATA_SPLIT)
    output_label_dir = os.path.join(OUTPUT_DIR, "labels", DATA_SPLIT)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    print("Copying selected images and labels...")
    for basename in tqdm(files_to_copy):
        # Find the correct image file extension (.jpg, .png, etc.)
        image_file_found = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_image_path = os.path.join(image_dir, basename + ext)
            if os.path.exists(potential_image_path):
                image_file_found = potential_image_path
                break
        
        label_file = basename + '.txt'
        source_label_path = os.path.join(label_dir, label_file)
        
        if image_file_found and os.path.exists(source_label_path):
            shutil.copy2(image_file_found, output_image_dir)
            shutil.copy2(source_label_path, output_label_dir)

    # Bonus: Copy the original data.yaml file for convenience
    shutil.copy2(DATASET_PATH, OUTPUT_DIR)

    print(f"\n--- Done! ---")
    print(f"Copied {len(files_to_copy)} images and their labels to: {OUTPUT_DIR}")

if __name__ == "__main__":
    select_yolo_subset()
