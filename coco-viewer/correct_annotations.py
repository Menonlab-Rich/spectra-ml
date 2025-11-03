import json
import os

def correct_coco_segmentation_format(input_json_path, output_json_path):
    """
    Reads a COCO JSON file and corrects the segmentation format for each annotation.

    It changes the format from a flat list [x1, y1, ...] to the required
    nested list [[x1, y1, ...]].
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"🔄 Reading annotations from: {input_json_path}")
    
    # Open and load the original JSON file
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    annotations_corrected = 0
    # Iterate through each annotation in the list
    for annotation in coco_data.get('annotations', []):
        segmentation = annotation.get('segmentation')
        
        # Check if segmentation exists, is a list, and is not already nested
        # A simple way to check if it's a flat list is to see if the first element is a number
        if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], (int, float)):
            # Wrap the flat list in another list to correct the format
            annotation['segmentation'] = [segmentation]
            annotations_corrected += 1

    print(f"✅ Corrected {annotations_corrected} annotation(s).")

    # Save the modified data to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"💾 Corrected file saved successfully to: {output_json_path}")


# --- HOW TO USE ---
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Correct COCO annotations")
    parser.add_argument("input_annotations", help="path to the input annotations")
    parser.add_argument("output_annotations", help="Path to the output annotations")
    args = parser.parse_args()


    # 3. Run the correction function
    correct_coco_segmentation_format(args.input_annotations, args.output_annotations)
