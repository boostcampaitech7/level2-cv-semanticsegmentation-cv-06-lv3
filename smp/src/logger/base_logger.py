from src.utils.data_utils import load_yaml_config

import os
import yaml

def log_configs_to_file(config, output_dir):
    """
    설정 내용을 텍스트 파일로 저장.
    
    Args:
        config (dict): 전체 설정을 포함한 딕셔너리.
    """
    output_dir = f'./logs/{output_dir[0]}/{output_dir[1]}'

    os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리 생성

    # 파일 저장 함수
    def save_to_file(content, filename):
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(content)

    # 데이터 구성 저장
    data_config = load_yaml_config(config["path"]["data"])
    save_to_file(yaml.dump(data_config), "data_config.txt")

    # 증강 구성 저장
    augmentation_config = load_yaml_config(config["path"]["augmentation"])
    save_to_file(yaml.dump(augmentation_config), "augmentation_config.txt")

    # 모델 구성 저장
    model_config = load_yaml_config(config["path"]["model"])
    save_to_file(yaml.dump(model_config), "model_config.txt")

    # 학습 설정 저장
    train_config = load_yaml_config(config["path"]["train"])
    save_to_file(yaml.dump(train_config), "train_config.txt")

    # 손실함수 설정 저장
    loss_config = load_yaml_config(config["path"]["loss"])
    save_to_file(yaml.dump(loss_config), "loss_config.txt")

    # 유틸 설정 저장
    util_config = load_yaml_config(config["path"]["loss"])
    save_to_file(yaml.dump(util_config), "util_config.txt")



