import hydra
from transformers import YolosImageProcessor
from setup import OmegaConf, DictConfig
from model import YolosLitModule
from data import YolosDataModule
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
# --- 3. The Main Training Function ---
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))

    processor_name = cfg.mode.pretrained_model_name or "hustvl/yolos-tiny"
    image_processor = YolosImageProcessor.from_pretrained(processor_name)
    
    data_module = YolosDataModule(cfg, image_processor)
    model_module = YolosLitModule(cfg)

    # ✅ THIS IS THE CALLBACK YOU ASKED FOR
    # It monitors the 'map' metric from the validation epoch end.
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.output_dir,
        filename='yolos-{epoch:02d}-{map:.4f}',
        monitor='map', # This key comes from the COCO metric
        mode='max',
        save_top_k=3,
    )
    
    trainer = L.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback], # Add the callback here
        logger=TensorBoardLogger(f"{cfg.training.output_dir}/logs/"),
        log_every_n_steps=10,
    )

    print("\n🔥 Starting training with PyTorch Lightning... 🔥")
    trainer.fit(model_module, datamodule=data_module)


if __name__ == "__main__":
    main()
