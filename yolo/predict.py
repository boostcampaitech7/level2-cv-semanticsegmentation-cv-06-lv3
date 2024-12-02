import os

import pandas as pd
import torch
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

PRETRAIND_WEIGHTS = "/home/taeyoung4060ti/바탕화면/부스트코스/level2-cv-semanticsegmentation-cv-06-lv3/yolo/yolo-seg/yolo9e-seg/weights/best.pt"

CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# Load a model
model = YOLO(PRETRAIND_WEIGHTS)
model.to("cuda:0")

base_dir = "YOLO_data/images/test"
rles = []
filename_and_class = []
for path in tqdm(os.listdir(base_dir)):
    img_path = os.path.join(base_dir, path)

    results = model.predict(
        source=img_path,
        imgsz=1024,
        project="yolo-seg",
        save=False,
        name="yolo11n-seg",
        retina_masks=True,
    )

    outputs = results[0].masks.data.unsqueeze(0)
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).detach().cpu().numpy()

    best_detections = {}

    for i, conf in enumerate(results[0].boxes.conf):
        cls = int(results[0].boxes.cls[i].item())  # 클래스 인덱스
        class_name = IND2CLASS[cls]

        # 해당 클래스에 대해 더 높은 신뢰도의 감지라면 업데이트
        if (
            class_name not in best_detections
            or conf > best_detections[class_name]["conf"]
        ):
            best_detections[class_name] = {
                "conf": conf,
                "mask": outputs[0][i],  # 해당 마스크 저장
            }

    # 최고 신뢰도의 감지만 RLE 인코딩 및 결과 저장
    for class_name, data in best_detections.items():
        rle = encode_mask_to_rle(data["mask"])
        rles.append(rle)
        filename_and_class.append(f"{class_name}_{path}")


# CLASSES 순서대로 정렬하기 위한 클래스와 파일 이름 추출
classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]

# 데이터프레임 생성
df = pd.DataFrame(
    {
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    }
)

# CLASSES 순서로 정렬
df["class_order"] = df["class"].map({cls: i for i, cls in enumerate(CLASSES)})
df = df.sort_values(by=["image_name", "class_order"]).drop(columns="class_order")

# 결과 저장
os.makedirs("./outputs", exist_ok=True)
df.to_csv("./outputs/output.csv", index=False)
