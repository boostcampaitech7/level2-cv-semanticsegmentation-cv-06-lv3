from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

def _get_logger(config):
    logger = []
    if config["use_tensorboard"]:
        logger.append(
            TensorBoardLogger(f'./logs/{config["name"]}')
        )
    if config["use_wandb"]:
        logger.append(
            WandbLogger(
                project=config["project"],
                name=config["name"]
            )
        )
    
    return logger