import unittest
import torch
from src.models.smp_model import SmpModel
from src.utils.data_utils import load_yaml_config
from src.plmodules.smp_module import SmpModule
from torch.utils.data import DataLoader, TensorDataset

# 테스트할 SmpModule 클래스
class TestSmpModule(unittest.TestCase):

    def setUp(self):
        # 테스트를 위한 샘플 config 파일을 mock으로 생성
        self.train_config_path = "/data/ephemeral/hbis/configs/train_configs/base.yaml"
        self.model_config_path = "/data/ephemeral/hbis/configs/model_configs/unet_efficientnet.yaml"

        # 모델 인스턴스를 생성
        self.smp_module = SmpModule(self.train_config_path, self.model_config_path)

        # 더미 데이터셋을 만들고 DataLoader로 감싸기
        self.batch_size = 8
        self.images = torch.randn(self.batch_size, 3, 256, 256)  # 예시 이미지
        self.labels = torch.randint(0, 2, (self.batch_size, 29, 256, 256)).float()  # 이진 마스크 예시

        dataset = TensorDataset(self.images, self.labels)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size)

    def test_training_step(self):
        # training_step이 제대로 동작하는지 테스트
        batch = (self.images, self.labels)
        outputs = self.smp_module.training_step(batch, batch_idx=0)

        # outputs는 loss 값이어야 함
        self.assertIsInstance(outputs, torch.Tensor)

    def test_validation_step(self):
        # validation_step이 제대로 동작하는지 테스트
        batch = (self.images, self.labels)
        dices = self.smp_module.validation_step(batch, batch_idx=0)

        # dice score는 텐서여야 함
        self.assertIsInstance(dices, torch.Tensor)

    def test_on_validation_epoch_end(self):
        # on_validation_epoch_end가 정상적으로 작동하는지 테스트
        mock_outputs = [torch.randn(self.batch_size) for _ in range(self.batch_size)]  # 모의 dice scores
        avg_dice = self.smp_module.on_validation_epoch_end(mock_outputs)

        # avg_dice는 float 값이어야 함
        self.assertIsInstance(avg_dice, float)

    def test_configure_optimizers(self):
        # configure_optimizers가 optimizer를 제대로 반환하는지 테스트
        optimizers, _ = self.smp_module.configure_optimizers()

        # optimizers는 리스트여야 하며, 첫 번째 요소는 Adam 옵티마이저이어야 함
        self.assertIsInstance(optimizers, list)
        self.assertEqual(len(optimizers), 1)
        self.assertIsInstance(optimizers[0], torch.optim.Adam)

    def test_dice_coef(self):
        # dice_coef가 제대로 작동하는지 테스트
        outputs = torch.randn(self.batch_size, 1, 256, 256).sigmoid()
        targets = torch.randint(0, 2, (self.batch_size, 1, 256, 256)).float()

        dice_score = self.smp_module.dice_coef(outputs, targets)

        # dice_score는 텐서여야 함
        self.assertIsInstance(dice_score, torch.Tensor)

        # dice_score는 각 배치에 대해 하나의 값이 나와야 함
        self.assertEqual(dice_score.shape, (self.batch_size,))

if __name__ == '__main__':
    unittest.main()
