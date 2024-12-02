import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.utils.data_utils import load_yaml_config
from src.data.custom_datamodules.xray_datamodule import XRayDataModule
from src.plmodules.smp_module import SmpModule

import torch

def main(config_path, weights, use_crf=False):

    print(f"... Loading config from {config_path} ...")

    config = load_yaml_config(config_path=config_path)
    data_config = load_yaml_config(config_path=config["path"].get("data", None))

    print(f"... {config_path} file loaded ...")

    if use_crf:
        print("... Use CRF postprocessing ...")

    data_module = XRayDataModule(
        config=config
    )

    data_module.setup()
    
    # subset = torch.utils.data.Subset(data_module.inference_dataset, range(5))
    
    dataloader = DataLoader(
            data_module.inference_dataset,
            batch_size=data_config["data"]["train"].get("batch_size", 2),
            num_workers=data_config["data"]["train"].get("num_workers", 4),
            shuffle=True,
            persistent_workers=True,
        )
    
    model = SmpModule(
        config=config
    )


    trainer = pl.Trainer(
        accelerator='gpu',
    )

    metrics = trainer.test(
        model=model,
        dataloaders=dataloader,
        ckpt_path=weights
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to the weights"
    )
    parser.add_argument(
        "--use_crf", action='store_true', help="Use CRF postprocessing"
    )
    args = parser.parse_args()
    main(args.config, args.weights, args.use_crf)



