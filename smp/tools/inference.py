import argparse
import pytorch_lightning as pl

from src.utils.data_utils import load_yaml_config
from src.data.custom_datamodules.xray_datamodule import XRayDataModule
from src.plmodules.smp_module import SmpModule

def main(config_path, weights):
    print(f"Loading config from {config_path} ...")
    config = load_yaml_config(config_path=config_path)
    print(f"Completed!")

    # load datamodule
    data_module = XRayDataModule(config=config)

    # setup datamodule
    data_module.setup()

    # load model
    model = SmpModule(config=config)

    # load trainer
    trainer = pl.Trainer(
        accelerator='gpu',
    )

    metrics = trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path=weights
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to a checkpoint file"
    )
    args = parser.parse_args()
    main(args.config, args.weights)



