from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

def _get_callbacks(config, logger_config):
    callbacks = []

    callbacks.append(
        LearningRateMonitor(logging_interval='epoch')
    )

    if config.get("model_checkpoint", {}).get("enabled", False):
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"./logs/{logger_config['project']}/{logger_config['name']}/checkpoints/",
                filename="{epoch:02d}-{avg_dice_score:.3f}",
                **config["model_checkpoint"]["params"]
            )
        )

    if config.get("earlystopping", {}).get("enabled", False):
        callbacks.append(
            EarlyStopping(
                **config["earlystopping"]["params"]
            )
        )

    return callbacks