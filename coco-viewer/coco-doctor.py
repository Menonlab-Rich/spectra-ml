import json
import argparse

def diagnose_coco(json_path):
    """
    Reads a COCO JSON file and checks for common issues that cause
    conversion tools to fail silently.
    """
    print(f"🩺 Starting diagnosis of {json_path}...\n")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ CRITICAL: Could not read or parse the JSON file. Error: {e}")
        return

    # --- Check 1: Ensure essential keys exist ---
    essential_keys = ['images', 'annotations', 'categories']
    missing_keys = [key for key in essential_keys if key not in data]
    if missing_keys:
        print(f"❌ CRITICAL: The JSON is missing essential keys: {', '.join(missing_keys)}")
        return

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    print(f"✅ Found {len(images)} images.")
    print(f"✅ Found {len(annotations)} annotations.")
    print(f"✅ Found {len(categories)} categories.")
    
    if not all([images, annotations, categories]):
        print("\n❌ CRITICAL: One of the essential sections (images, annotations, or categories) is empty.")
        return

    # --- Check 2: Create lookups for efficient checking ---
    image_ids = {img['id'] for img in images}
    category_ids = {cat['id'] for cat in categories}
    
    # --- Check 3: Validate annotations ---
    print("\n🧐 Analyzing annotations...")
    valid_annotations = 0
    errors = {
        'missing_image_id': 0,
        'invalid_category_id': 0,
        'invalid_segmentation': 0,
        'zero_area': 0
    }

    for i, ann in enumerate(annotations):
        is_valid = True
        
        # Check if annotation's image_id exists in the images list
        if ann.get('image_id') not in image_ids:
            errors['missing_image_id'] += 1
            is_valid = False

        # Check if annotation's category_id exists in the categories list
        if ann.get('category_id') not in category_ids:
            errors['invalid_category_id'] += 1
            is_valid = False
        
        # Check if segmentation is a non-empty list
        segmentation = ann.get('segmentation', [])
        if not isinstance(segmentation, list) or not segmentation:
            errors['invalid_segmentation'] += 1
            is_valid = False
        elif isinstance(segmentation[0], list) and (len(segmentation[0]) < 6 or len(segmentation[0]) % 2 != 0):
             # Check if it's a valid polygon (at least 3 points)
            errors['invalid_segmentation'] += 1
            is_valid = False

        # Check if area is greater than zero
        if ann.get('area', 0) <= 0:
            errors['zero_area'] += 1
            is_valid = False
            
        if is_valid:
            valid_annotations += 1

    print("\n--- Diagnosis Report ---")
    if valid_annotations == len(annotations):
        print("\n🎉 SUCCESS: All annotations appear to be valid!")
        print("This suggests the problem might be with the Ultralytics converter or its dependencies.")
    else:
        print(f"\n⚠️ Found issues in {len(annotations) - valid_annotations} out of {len(annotations)} annotations:")
        if errors['missing_image_id'] > 0:
            print(f"  - {errors['missing_image_id']} annotations point to an `image_id` that doesn't exist.")
        if errors['invalid_category_id'] > 0:
            print(f"  - {errors['invalid_category_id']} annotations use a `category_id` that isn't defined.")
        if errors['invalid_segmentation'] > 0:
            print(f"  - {errors['invalid_segmentation']} annotations have malformed or empty `segmentation` data.")
        if errors['zero_area'] > 0:
            print(f"  - {errors['zero_area']} annotations have an area of 0, which converters often ignore.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnose a COCO JSON annotation file.")
    parser.add_argument('json_path', type=str, help='Path to the COCO JSON file to diagnose.')
    args = parser.parse_args()
    diagnose_coco(args.json_path)
