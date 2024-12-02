# sliding_window.py
import torch

class SlidingWindowInference:
    def __init__(self, model, patch_size, stride, batch_size, num_classes):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_classes = num_classes
    
    def _sliding_window_step(self, batch):
        if len(batch) == 2:
            images, labels = batch
        else:
            images = batch
            labels = None
        
        stride = self.stride
        patch_size = self.patch_size
        batch_patches = []
        batch_coords = []
        batch_size = self.batch_size

        _, _, H, W = images.shape  # 입력 이미지 크기 (Batch, Channels, Height, Width)
        outputs_full = torch.zeros((images.size(0), self.num_classes, H, W)).to(images.device)

        # 슬라이딩 윈도우로 이미지 나누기
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = images[:, :, i:i + patch_size, j:j + patch_size]
                batch_patches.append(patch)
                batch_coords.append((i, j))

                if len(batch_patches) == batch_size:
                    batch_patches_tensor = torch.cat(batch_patches, dim=0)
                    batch_outputs = self.model(batch_patches_tensor)
                    for k, (x, y) in enumerate(batch_coords):
                        outputs_full[:, :, x:x + patch_size, y:y + patch_size] += batch_outputs[k]
                    batch_patches = []
                    batch_coords = []

        if batch_patches:
            batch_patches_tensor = torch.cat(batch_patches, dim=0)
            batch_outputs = self.model(batch_patches_tensor)
            for k, (x, y) in enumerate(batch_coords):
                outputs_full[:, :, x:x + patch_size, y:y + patch_size] += batch_outputs[k]

        norm_map = torch.zeros_like(outputs_full)
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                norm_map[:, :, i:i + patch_size, j:j + patch_size] += 1
        outputs_full /= norm_map

        if labels is not None:
            loss = self.model.criterion(outputs_full, labels)
            return outputs_full, loss

        return outputs_full
