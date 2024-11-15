from src.utils.data_utils import load_yaml_config

def log_configs_to_tensorboard(logger, config):
    # 데이터 구성 로깅
    data_config = load_yaml_config(config["path"]["data"])
    logger.experiment.add_text("Data Configurations", str(data_config))
    
    # 증강 구성 로깅
    augmentation_config = load_yaml_config(config["path"]["augmentation"])
    logger.experiment.add_text("Augmentation Configurations", str(augmentation_config))
    
    # 모델 구성 로깅
    model_config = load_yaml_config(config["path"]["model"])
    logger.experiment.add_text("Model Configurations", str(model_config))
    
    # 옵티마이저 설정 로깅
    logger.experiment.add_text("Optimizer Configurations", str(config["optimizer"]))
    
    # 스케줄러 설정 로깅
    logger.experiment.add_text("Scheduler Configurations", str(config["scheduler"]))


