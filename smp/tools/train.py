import argparse
import pytorch_lightning as pl

from src.logger.custom_logger import _get_logger
from src.logger.base_logger import log_configs_to_tensorboard
from src.callbacks.base_callback import _get_callbacks
from src.utils.data_utils import load_yaml_config
from src.data.custom_datamodules.xray_datamodule import XRayDataModule
from src.plmodules.smp_module import SmpModule

def main(config_path):
    print(f"Loading config from {config_path} ...")
    config = load_yaml_config(config_path=config_path)
    print(f"Completed!")

    aug_path = config["path"]["augmentation"]
    data_path = config["path"]["data"]
    model_path = config["path"]["model"]

    data_module = XRayDataModule(
        data_config_path=data_path,
        augmentation_config_path=aug_path
    )

    data_module.setup()

    model = SmpModule(
        train_config_path=config_path,
        model_config_path=model_path
    )

    logger = _get_logger(
        config["logger"]
    )

    log_configs_to_tensorboard(logger[0], config)

    callbacks = _get_callbacks(
        config["callbacks"],
        config["logger"]
    )

    trainer = pl.Trainer(
        **config["trainer"],
        callbacks=callbacks,
        logger=logger
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    main(args.config)



