from omegaconf import DictConfig 
from torchmetrics.detection import MeanAveragePrecision
import lightning as pl
import torch
from ultralytics import YOLO
from transformers import (
    YolosConfig,
    YolosForObjectDetection,
    YolosImageProcessor,
    get_scheduler,
)

# --- 2. The LightningModule: Defines the model and the training loop ---
class YolosLitModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg) # Saves cfg to self.hparams
        self.cfg = cfg
        self._build_model()

        # For post-processing and metrics
        # Use a real model name to ensure processor is configured correctly
        processor_name = self.cfg.mode.pretrained_model_name or "hustvl/yolos-tiny"
        self.image_processor = YolosImageProcessor.from_pretrained(processor_name)
        self.val_metric = MeanAveragePrecision(box_format='xyxy')

    def _build_model(self):
        # Initialize model based on the chosen mode
        if self.cfg.mode.from_scratch:
            print("\n🚀 Initializing model from scratch...")
            model_config = YolosConfig(num_labels=self.cfg.model.num_labels)
            self.model = YolosForObjectDetection(config=model_config)
        else:
            print(f"\n🚀 Loading pretrained model '{self.cfg.mode.pretrained_model_name}'...")
            self.model = YolosForObjectDetection.from_pretrained(
                self.cfg.mode.pretrained_model_name,
                num_labels=self.cfg.model.num_labels,
                ignore_mismatched_sizes=True,
            )

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.weight_decay)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
        )

        scheduler = { "scheduler": scheduler, "interval": "step", "frequency": 1 }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # The model returns a dictionary of outputs
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self(pixel_values=batch["pixel_values"])
        target_sizes = ([label["size"] for label in labels])
        predictions = self.image_processor.post_process_object_detection(
                outputs, threshold=0.0, target_sizes=target_sizes
        )

        # Format for the metric
        preds = [
            {"boxes": pred["boxes"], "scores": pred["scores"], "labels": pred["labels"]} 
            for pred in predictions
        ]
        targets = [
            {"boxes": label["boxes"], "labels": label["class_labels"]} 
            for label in labels
        ]
        
        self.val_metric.update(preds, targets)
        
    def on_validation_epoch_end(self):
        # Compute and log the final metrics
        metrics = self.val_metric.compute()
        self.log_dict({m_key: m_value for m_key, m_value in metrics.items() if m_key != 'classes'}, prog_bar=True)


class YoloV8LitModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg) # Saves cfg to self.hparams
        self.cfg = cfg
        self._build_model()

        # For post-processing and metrics
        # Use a real model name to ensure processor is configured correctly
        self.val_metric = MeanAveragePrecision(box_format='xyxy')

    def _build_model(self):
        # Initialize model based on the chosen mode
        if self.cfg.mode.from_scratch:
            print("\n🚀 Initializing model from scratch...")
            self.model = YOLO(self.cfg.model.model_config)
        else:
            print(f"\n🚀 Loading pretrained model '{self.cfg.mode.pretrained_model_name}'...")
            self.model = YolosForObjectDetection.from_pretrained(
                self.cfg.mode.pretrained_model_name,
                num_labels=self.cfg.model.num_labels,
                ignore_mismatched_sizes=True,
            )

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.weight_decay)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
        )

        scheduler = { "scheduler": scheduler, "interval": "step", "frequency": 1 }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # The model returns a dictionary of outputs
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        outputs = self(pixel_values=batch["pixel_values"])
        target_sizes = ([label["size"] for label in labels])
        predictions = self.image_processor.post_process_object_detection(
                outputs, threshold=0.0, target_sizes=target_sizes
        )

        # Format for the metric
        preds = [
            {"boxes": pred["boxes"], "scores": pred["scores"], "labels": pred["labels"]} 
            for pred in predictions
        ]
        targets = [
            {"boxes": label["boxes"], "labels": label["class_labels"]} 
            for label in labels
        ]
        
        self.val_metric.update(preds, targets)
        
    def on_validation_epoch_end(self):
        # Compute and log the final metrics
        metrics = self.val_metric.compute()
        self.log_dict({m_key: m_value for m_key, m_value in metrics.items() if m_key != 'classes'}, prog_bar=True)
