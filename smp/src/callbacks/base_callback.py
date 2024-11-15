from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

def _get_callbacks(config, logger_config):
    callbacks = []

    callbacks.append(
        LearningRateMonitor(logging_interval='epoch')
    )

    if config["model_checkpoint"]:
        callbacks.append(
            ModelCheckpoint(
                **config["model_checkpoint_params"],
                dirpath=f"./logs/{logger_config['name']}/checkpoints/",
                filename="{epoch:02d}-{val_loss:.2f}",
            )
        )

    if config["earlystopping"]:
        callbacks.append(
            EarlyStopping(
                **config["earlystopping_params"]
            )
        )

    return callbacks