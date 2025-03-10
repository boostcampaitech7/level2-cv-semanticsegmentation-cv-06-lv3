import unittest
from torch.utils.data import DataLoader
from src.data.custom_datamodules.xray_datamodule import XRayDataModule
import albumentations as A

class TestXRayDataModule(unittest.TestCase):
    def setUp(self):
        # 테스트 설정 파일 경로 제공
        self.data_config_path = "/data/ephemeral/hbis/configs/data_configs/xray_data.yaml"
        self.augmentation_config_path = "/data/ephemeral/hbis/configs/augmentation_configs/base_augmentation.yaml"
        
        # XRayDataModule 인스턴스 생성
        self.data_module = XRayDataModule(
            data_config_path=self.data_config_path,
            augmentation_config_path=self.augmentation_config_path,
            seed=42
        )
        
    def test_setup(self):
        # setup 메서드 호출하여 데이터셋 로드
        self.data_module.setup(stage="fit")
        
        # 데이터셋이 제대로 생성되었는지 확인
        self.assertIsNotNone(self.data_module.train_dataset)
        self.assertIsNotNone(self.data_module.val_dataset)
        self.assertIsNotNone(self.data_module.test_dataset)
    
    def test_augmentation_transforms(self):
        # setup 호출 후 augmentation 여부를 확인
        self.data_module.setup(stage="fit")
        train_transforms = self.data_module.train_dataset.transforms

        # augmentation_config 설정에 따라 필요한 변환이 있는지 확인
        self.assertTrue(any(isinstance(t, A.Resize) for t in train_transforms.transforms))
    
    def test_dataloaders(self):
        # 데이터로더 테스트
        self.data_module.setup(stage="fit")
        
        # train_dataloader 확인
        train_loader = self.data_module.train_dataloader()
        self.assertIsInstance(train_loader, DataLoader)
        
        # batch_size와 num_workers가 config와 일치하는지 확인
        self.assertEqual(train_loader.batch_size, self.data_module.config["data"]["train"]["batch_size"])
        self.assertEqual(train_loader.num_workers, self.data_module.config["data"]["train"]["num_workers"])

        # 첫 번째 배치 불러오기
        for batch in train_loader:
            images, labels = batch
            self.assertEqual(images.shape[0], self.data_module.config["data"]["train"]["batch_size"])
            break

    def test_inference_dataloader(self):
        # Inference용 데이터로더가 설정 파일과 맞는지 테스트
        self.data_module.setup(stage="test")
        test_loader = self.data_module.test_dataloader()

        # test_loader가 DataLoader 인스턴스인지 확인
        self.assertIsInstance(test_loader, DataLoader)
        # inference 모드이므로 label이 없는지 확인 (이미지 데이터만 있어야 함)
        for batch in test_loader:
            images, image_names = batch
            self.assertEqual(images.shape[0], self.data_module.config["data"]["test"]["batch_size"])
            break

if __name__ == "__main__":
    unittest.main()
