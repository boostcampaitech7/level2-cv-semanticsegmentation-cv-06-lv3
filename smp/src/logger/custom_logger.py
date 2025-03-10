from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

def _get_logger(config):
    logger = []
    if config.get("tensorboard", False):
        logger.append(
            TensorBoardLogger(
                save_dir=f'./logs',
                name=config["project"],
                version=config["name"]
                )
        )
    if config.get("wandb", False):
        logger.append(
            WandbLogger(
                project=config["project"],
                name=config["name"]
            )
        )
    
    return logger