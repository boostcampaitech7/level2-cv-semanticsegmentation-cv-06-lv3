import unittest
import os
from torch.utils.data import DataLoader
from src.data.datasets.xray_dataset import XRayDataset
from src.data.datasets.xray_inferencedataset import XRayInferenceDataset

class TestXRayDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 데이터 경로 설정
        cls.data_root = "/data/ephemeral/data"
        cls.image_path = os.path.join(cls.data_root, "train/DCM")
        cls.label_path = os.path.join(cls.data_root, "train/outputs_json")
    
    def test_xray_train_dataset(self):
        # XRayDataset 초기화 (훈련용)
        dataset = XRayDataset(image_path=self.image_path, label_path=self.label_path, is_train=True)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # 데이터 로딩 및 배치 모양 확인
        for images, labels in dataloader:
            self.assertEqual(len(images), 4, "배치 크기는 4여야 합니다.")
            self.assertEqual(images.shape[1], 3, "이미지는 3채널이어야 합니다.")
            # 예상하는 라벨 형식에 맞는지 확인
            self.assertEqual(len(labels), 4, "라벨 배치 크기는 4여야 합니다.")
            break  # 첫 번째 배치로 테스트 후 종료

    def test_xray_inference_dataset(self):
        # XRayInferenceDataset 초기화 (추론용)
        dataset = XRayInferenceDataset(image_path=self.image_path)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        # 데이터 로딩 및 배치 모양 확인
        for images, names in dataloader:
            self.assertEqual(len(images), 4, "배치 크기는 4여야 합니다.")
            self.assertEqual(images.shape[1], 3, "이미지는 3채널이어야 합니다.")
            self.assertEqual(len(names), 4, "이름 배치 크기는 4여야 합니다.")
            break  # 첫 번째 배치로 테스트 후 종료

if __name__ == "__main__":
    unittest.main()
