# ------------------------------------------------------------------
# train.py
# ------------------------------------------------------------------
import argparse
import torch
import torch.nn.functional as F
from ultralytics.models import YOLO, yolo
import albumentations as A
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import VarifocalLoss, v8SegmentationLoss
from ultralytics.utils.tal import  make_anchors
from typing import Any
from pathlib import Path

class CustomV8Loss(v8SegmentationLoss):
    def __init__(self, model):
        super().__init__(model)
        self.varifocal_loss = VarifocalLoss()
        self.class_weights = torch.tensor([1.0, 5.0], device=self.device)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            weight = self.class_weights[target_labels[fg_mask].long()]
            loss[2] = (self.varifocal_loss(pred_scores, target_scores, target_labels) * weight).sum() / target_scores_sum  # VFL way

            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl)

class CustomSegmentationModel(SegmentationModel):
    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)

    def init_criterion(self):
        return CustomV8Loss(self)

class CustomYOLO(YOLO):
    def __init__(self, model: str | Path = "yolo11n.pt", task: str | None = None, verbose: bool = False):
        super().__init__(model, task, verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        task_map = super().task_map
        task_map['segment'] = { "model": CustomSegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            }
        return task_map

transform = A.Compose([
    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.ShiftScaleRotate(p=0.5, rotate_limit=15, scale_limit=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def train_yolo(model_path, data_path, epochs, batch_size, img_size, project_name, run_name, patience):
    """
    Trains a YOLOv8 instance segmentation model.

    Args:
        model_path (str): Path to the pre-trained model file (e.g., 'yolov8n-seg.pt').
        data_path (str): Path to the dataset's .yaml file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        img_size (int): Input image size.
        project_name (str): Name of the project directory for saving results.
        run_name (str): Name of the specific run directory.
    """
    print("🧠 Loading pre-trained model...")
    # Load a pre-trained YOLOv8 segmentation model
    model = CustomYOLO(model_path)

    print(f"🚀 Starting training on '{data_path}' for {epochs} epochs...")
    
    # Train the model
    # The 'task' is automatically inferred as 'segment' from the model type (e.g., yolov8n-seg.pt)
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_name,
        name=run_name,
        patience=patience,
        # --- Augmentation Strategy ---
        # Disable mosaic
        mosaic=0.0,

        # Aggressive geometric transforms
        degrees=20,
        translate=0.2,
        scale=0.9,
        shear=2.0,
        perspective=0.01,
        flipud=0.3,
        fliplr=0.3,
        
        # Context-breaking transform
        copy_paste=0.1,

        # Moderate color-space transforms
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,

        mixup=0.01,
        cls=2,
    )
    
    print(f"✅ Training complete! Results saved to '{results.save_dir}'")

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Train a YOLOv8 Instance Segmentation Model")
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt', help='Path to the pre-trained model (e.g., yolov8n-seg.pt, yolov8s-seg.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to the data configuration (.yaml) file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=float, default=8, help='Batch size (adjust based on your GPU memory)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--project', type=str, default='runs/segment', help='Project name to save results under')
    parser.add_argument('--name', type=str, default='train', help='Specific run name (e.g., exp1, run_with_more_data)')
    parser.add_argument('--patience', type=int, default=20, help='Stop training after n epochs of no improvement')

    args = parser.parse_args()

    # Call the training function
    train_yolo(
        model_path=args.model,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        project_name=args.project,
        run_name=args.name,
        patience=args.patience,
    )
