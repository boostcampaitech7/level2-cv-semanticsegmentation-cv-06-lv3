# ✨ Segmentation Models Pytorch ✨

> smp for xray datasets (handbone segmentation task)

## ❓ 사용 방법

#### 폴더 이동
```bash
cd smp
```

#### 종속 라이브러리 설치
```bash
pip install -r requirements.txt
```

#### config file 수정

1. data_configs
    - dataset path 설정
    - train, val, test batch_size, num_workers 설정
    - seed 설정

2. augmentation_configs
    - configs 파일 생성해서 실험하는 편이 좋음
    - use_augmentation: False
        - resize(512, 512)만 적용 -- baseline
    
    - use_augmentation: True
        - albumentation 증강 기법 적용 가능
        - 일단 색 관련 변환만
        - 기하학적 변환은 테스팅 아직 안해봄 (mask 잘 바뀌는지 확인 필요 resize 되는거보면 아마 잘 바뀌긴 할 듯)

3. model_configs
    - configs 파일 생성해서 실험하는 편이 좋음
    - model
        - documentation 참조
        - arch
        - encoder_name
        - encoder_weights
        - in_channels
        - classes
    - loss
        - use_different_loss: True
            - default loss: BCE
        - use_different_loss: False
            - documentation 참조

4. train_configs
    - path 지정 (위에 3개의 configs path)
    - optimizer 수정 가능
    - lr_scheduler 수정 가능
    - logger (tensorboard, wandb 구현)
        - wandb 무슨 권한 문제 때문에 아직 테스팅은 안해봄

#### help 명령어 사용

```bash
python tools/train.py --help
```

#### 학습 시작

```bash
python tools/train.py --config={config_path}
```

#### tensorboard 로깅 확인

```bash
tensorboard --logdir logs
```

- 보면 임시로 text에 configs 관련 로깅도 넣어둠
- 현재 5epoch 당 validation 진행, 검증 손실 최솟값 기준 3개 모델 가중치 저장 중

