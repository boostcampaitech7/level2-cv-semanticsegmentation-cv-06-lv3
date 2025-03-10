import argparse
import pytorch_lightning as pl

from src.logger.custom_logger import _get_logger
from src.logger.base_logger import log_configs_to_file
from src.callbacks.base_callback import _get_callbacks
from src.utils.data_utils import load_yaml_config
from src.data.custom_datamodules.xray_datamodule import XRayDataModule
from src.plmodules.smp_module import SmpModule

def main(config_path, checkpoint_path=None):

    # load config file
    print(f"... Loading config from {config_path} ...")
    config = load_yaml_config(config_path=config_path)
    experiments_config = load_yaml_config(config["path"].get("experiments", ""))
    train_config = load_yaml_config(config["path"].get("train", ""))

    print(f"... {config_path} file loaded ...")

    # check whether auto mixed precision is used
    if "amp" in experiments_config and experiments_config["amp"].get("enabled", False):
        print(f'... Use auto mixed precision (amp) ...')
        amp = "16"
    else:
        amp = None

    # Load datamodule
    data_module = XRayDataModule(config=config)

    # Setup datamodule
    data_module.setup()

    # Load model
    if checkpoint_path:
        print(f"... Loading model from checkpoint: {checkpoint_path} ...")
        model = SmpModule.load_from_checkpoint(checkpoint_path, config=config)
    else:
        print(f"... Initializing a new model ...")
        model = SmpModule(config=config)
    
    # Setup logger
    logger = _get_logger(config=train_config.get("logger", {}))

    # log your setting in txt
    log_configs_to_file(
        config=config, 
        output_dir=[
            train_config["logger"].get("project", ""),
            train_config["logger"].get("name", "")
            ]
        )

    # Setup callback
    callbacks = _get_callbacks(
        train_config.get("callbacks", {}),
        train_config.get("logger", {}),
    )

    # Load trainer with your settings
    # callbacks: earlystop, val_interval
    # loggers: tensorboard, wandb
    # precision: use amp or not
    # you can use accumulate_grad_batches
    trainer = pl.Trainer(
        **train_config["trainer"],
        callbacks=callbacks,
        logger=logger,
        precision=amp,
        # accumulate_grad_batches=2,
    )

    # Fit your model
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to a config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to a checkpoint file (optional)"
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint)



