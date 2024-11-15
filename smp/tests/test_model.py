import unittest
import torch
from torch import nn
from src.utils.data_utils import load_yaml_config
from src.models.smp_model import SmpModel

def generate_dummy_labels(batch_size, num_classes, height, width, points):
    """
    points: list of points, each point is [x, y] or a list of [x, y] for each object in the image
    This function assumes the points are in the form [batch_size, num_points, 2]
    Returns: labels tensor in the shape [batch_size, num_classes, height, width]
    """
    labels = torch.zeros(batch_size, num_classes, height, width)
    
    for b in range(batch_size):
        for point, class_id in points[b]:  # points[b] contains list of points with class_id
            x, y = point
            labels[b, class_id, y, x] = 1  # (x, y) 위치에 해당 클래스의 마스크 설정

    return labels

class TestSmpModel(unittest.TestCase):
    def setUp(self):
        # 가상 설정 파일 경로, 실제 config 파일 경로 설정 필요
        self.config_path = "/data/ephemeral/hbis/configs/model_configs/unet_efficientnet.yaml"
        # 기본 설정 로딩
        self.config = load_yaml_config(self.config_path)
        # 모델 인스턴스 생성
        self.model = SmpModel(self.config_path)

    def test_forward_without_labels(self):
        """labels 없이 forward 메서드 테스트"""
        self.model.eval()

        # 입력 데이터 생성
        in_channels = self.config["model"]["in_channels"]
        num_classes = self.config["model"]["classes"]
        images = torch.rand(2, in_channels, 128, 128)  # 128x128 크기의 임의 이미지 배치

        # Forward 테스트 (labels 없이)
        with torch.no_grad():
            output = self.model(images)  # labels 인수 없이 호출

        # 출력 형태가 예상대로인지 확인
        self.assertEqual(output.shape, (2, num_classes, 128, 128))  # 예측 출력 차원 확인

    def test_forward_with_labels(self):
        """labels 포함한 forward 메서드 테스트 (출력과 손실 함께 반환)"""
        self.model.eval()

        # 입력 데이터 생성
        in_channels = self.config["model"]["in_channels"]
        num_classes = self.config["model"]["classes"]
        images = torch.rand(2, in_channels, 128, 128)

        # points로 dummy labels 생성
        points = [
            [[(10, 20), 0], [(50, 60), 1]],  # 이미지 1의 points
            [[(30, 40), 2], [(70, 80), 1]]   # 이미지 2의 points
        ]
        labels = generate_dummy_labels(2, num_classes, 128, 128, points)

        # Forward 테스트 (labels 포함)
        with torch.no_grad():
            output, loss = self.model(images, labels)  # labels 인수와 함께 호출

        # 출력 형태가 예상대로인지 확인
        self.assertEqual(output.shape, (2, num_classes, 128, 128))
        # 손실이 스칼라 값인지 확인
        self.assertTrue(isinstance(loss, torch.Tensor) and loss.dim() == 0)

if __name__ == "__main__":
    unittest.main()
