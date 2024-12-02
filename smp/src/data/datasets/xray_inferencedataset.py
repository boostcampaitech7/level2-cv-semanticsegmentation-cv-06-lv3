import os
import json
import numpy as np

import cv2
import torch
from torch.utils.data import Dataset

class XRayInferenceDataset(Dataset):
    def __init__(self, image_path, transforms=None):

        self.image_path = image_path

        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.image_path)
            for root, _dirs, files in os.walk(self.image_path)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }

        _filenames = pngs
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_path, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()

        return image, image_name