import argparse
from ultralytics import YOLO

def evaluate_model(model_path, data_path, img_size, project_name, run_name):
    """
    Evaluates a trained YOLOv8 instance segmentation model.

    Args:
        model_path (str): Path to the trained model weights (.pt file).
        data_path (str): Path to the dataset's .yaml file.
        img_size (int): Input image size for evaluation.
        project_name (str): Name for the project directory.
        run_name (str): Name for the specific evaluation run.
    """
    print(f"🧠 Loading trained model from: {model_path}")
    # Load the trained model
    model = YOLO(model_path)

    print(f"🧪 Evaluating model on dataset: {data_path}")
    
    # Evaluate the model on the validation set
    # The 'task' is automatically inferred from the model type
    metrics = model.val(
        conf=0.1,
        iou=0.6,
        data=data_path,
        imgsz=img_size,
        project=project_name,
        name=run_name,
        split='val'  # Specify you want to run on the validation set
    )
    
    print("✅ Evaluation complete! Metrics are printed above.")
    # The 'metrics' object contains detailed results if you need to access them programmatically
    # For example: print(metrics.box.map) for box mAP50-95

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model.")
    parser.add_argument('--model', type=str, required=True, help='Path to your trained .pt file.')
    parser.add_argument('--data', type=str, required=True, help='Path to your data configuration (.yaml) file.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for evaluation.')
    parser.add_argument('--project', type=str, default='runs/segment', help='Project name to save evaluation results.')
    parser.add_argument('--name', type=str, default='evaluate', help='Name for the evaluation run folder.')
    
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        img_size=args.imgsz,
        project_name=args.project,
        run_name=args.name
    )
