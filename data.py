import lightning as pl
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, Image
from omegaconf import DictConfig
import json
from collections import defaultdict

class YolosDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, image_processor):
        super().__init__()
        self.cfg = cfg
        self.image_processor = image_processor

    def setup(self, stage=None):
        with open(self.cfg.dataset.annotations_path, 'r') as f:
            coco_data = json.load(f)

        category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        annotations_grouped = defaultdict(list)
        for ann in coco_data['annotations']:
            annotations_grouped[ann['image_id']].append(ann)

        dataset_list = []
        for img_info in coco_data['images']:
            image_id = img_info['id']
            annotations = annotations_grouped[image_id]
            if not annotations:
                continue
            
            dataset_list.append({
                "image_id": image_id,
                "image": f"{self.cfg.dataset.image_directory}/{img_info['file_name']}",
                "width": img_info['width'],
                "height": img_info['height'],
                "objects": {
                    "id": [ann['id'] for ann in annotations],
                    "area": [ann['area'] for ann in annotations],
                    "bbox": [ann['bbox'] for ann in annotations],
                    "category": [category_id_to_name[ann['category_id']] for ann in annotations],
                    "category_id": [ann['category_id'] for ann in annotations], # ✅ FIX 1: Store the original category_id
                }
            })

        full_dataset = Dataset.from_list(dataset_list).cast_column("image", Image())
        split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
        self.train_dataset = split_dataset['train'].with_transform(self._transform)
        self.val_dataset = split_dataset['test'].with_transform(self._transform)

    def _transform(self, examples):
        images = [img.convert("RGB") for img in examples["image"]]
        annotations = []
        for i in range(len(examples['image_id'])):
            objects_for_image = examples['objects'][i]
            
            # ✅ FIX 2: Pass 'category_id' to the processor instead of 'category'
            ann_dicts = [
                {'id': id_val, 'area': area, 'bbox': bbox, 'category_id': cat_id}
                for id_val, area, bbox, cat_id in zip(
                    objects_for_image['id'],
                    objects_for_image['area'],
                    objects_for_image['bbox'],
                    objects_for_image['category_id'] # Use the category_id list
                )
            ]
            
            annotations.append({
                "image_id": examples['image_id'][i],
                "annotations": ann_dicts
            })

        inputs = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        inputs['labels'] = [l for l in inputs['labels']]
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        return inputs

    def _collate_fn(self, batch):
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        labels = [x["labels"] for x in batch]
        return {"pixel_values": pixel_values, "labels": labels}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, collate_fn=self._collate_fn, batch_size=self.cfg.training.batch_size, shuffle=True, num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, collate_fn=self._collate_fn, batch_size=self.cfg.training.batch_size, num_workers=4
        )
