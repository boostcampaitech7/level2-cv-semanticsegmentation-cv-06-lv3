# python native
import os
import argparse

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import albumentations as A
import ttach as tta

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.plmodules.smp_module import SmpModule
from src.utils.data_utils import load_yaml_config
from src.data.datasets.xray_inferencedataset import XRayInferenceDataset
from src.utils.constants import CLASSES, IND2CLASS

# 데이터 경로를 입력하세요

IMAGE_ROOT = "/home/taeyoung4060ti/바탕화면/부스트코스/level2-cv-semanticsegmentation-cv-06-lv3/data/test/DCM"


def test(model, data_loader, thr=0.5):
    transform = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Scale(scales=[1, 1.5, 2]),
        ]
    )

    model = model.to("cuda:0").eval()
    tta_model = tta.SegmentationTTAWrapper(model, transform, merge_mode="mean")

    rles = []
    filename_and_class = []
    with torch.no_grad():

        for step, (images, image_names) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images = images.to("cuda:0")
            outputs = tta_model(images)

            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = SmpModule.encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


def main(config_path, weights):
    print(f"Loading config from {config_path} ...")
    config = load_yaml_config(config_path=config_path)
    train_config = load_yaml_config(config["path"].get("train", ""))
    print(f"Completed!")

    tf = A.Resize(1024, 1024)

    test_dataset = XRayInferenceDataset(IMAGE_ROOT, transforms=tf)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    model = SmpModule.load_from_checkpoint(
        checkpoint_path=weights,
        config=config,
    )

    model = model.model

    rles, filename_and_class = test(model, test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )

    os.makedirs("tta_outputs", exist_ok=True)
    df.to_csv("tta_outputs/TTA_output.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to the weights"
    )
    args = parser.parse_args()
    main(args.config, args.weights)
