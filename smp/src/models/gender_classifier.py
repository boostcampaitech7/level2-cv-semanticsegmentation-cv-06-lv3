import torch
import torch.nn as nn

class GenderClassifier(nn.Module):
    def __init__(self, num_seg_classes=29, num_gender_classes=2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(num_seg_classes, 64, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_gender_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, masks, labels=None):
        outputs = self.classifier(masks)

        if labels is not None:
            loss = self.criterion(outputs, labels)

            return outputs, loss
        
        return outputs