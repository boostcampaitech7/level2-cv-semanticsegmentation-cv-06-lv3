import os
import json
import numpy as np
from sklearn.model_selection import GroupKFold

# 데이터 경로 설정
IMAGE_ROOT = "../data/train/DCM"
LABEL_ROOT = "../data/train/outputs_json"

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

# 이미지 및 JSON 파일 로드
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _, files in os.walk(IMAGE_ROOT)
    for fname in files
    if fname.lower().endswith(".png")
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _, files in os.walk(LABEL_ROOT)
    for fname in files
    if fname.lower().endswith(".json")
}

# 이미지와 라벨이 일치하는지 확인
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
assert pngs_fn_prefix == jsons_fn_prefix, "이미지와 라벨 파일이 일치하지 않습니다."

pngs = sorted(pngs)
jsons = sorted(jsons)

_filenames = np.array(pngs)
_labelnames = np.array(jsons)

groups = [os.path.dirname(fname) for fname in _filenames]
ys = np.zeros(len(_filenames))  # 더미 레이블

gkf = GroupKFold(n_splits=5)
train_indices = []
valid_indices = []

for i, (train_idx, valid_idx) in enumerate(gkf.split(_filenames, ys, groups)):
    if i == 0:
        valid_indices.extend(valid_idx)
    else:
        train_indices.extend(valid_idx)

train_filenames = _filenames[train_indices]
train_labelnames = _labelnames[train_indices]
valid_filenames = _filenames[valid_indices]
valid_labelnames = _labelnames[valid_indices]

data_root_dir = "YOLO_data"
os.makedirs(os.path.join(data_root_dir, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(data_root_dir, "labels", "val"), exist_ok=True)
os.makedirs(os.path.join(data_root_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(data_root_dir, "images", "val"), exist_ok=True)


def process_data(filenames, labelnames, split):
    for image_name, label_name in zip(filenames, labelnames):
        # 이미지 복사
        src_image_path = os.path.join(IMAGE_ROOT, image_name)
        dst_image_path = os.path.join(
            data_root_dir, "images", split, os.path.basename(image_name)
        )
        os.system(f"cp {src_image_path} {dst_image_path}")

        # 라벨 처리
        label_path = os.path.join(LABEL_ROOT, label_name)
        with open(label_path, "r") as f:
            label_json = json.load(f)

        label_output_path = os.path.join(
            data_root_dir,
            "labels",
            split,
            os.path.splitext(os.path.basename(label_name))[0] + ".txt",
        )
        with open(label_output_path, "w") as f_out:
            for annotation in label_json["annotations"]:
                points = annotation["points"]
                coords = [str(coord / 2048) for point in points for coord in point]
                temp = " ".join(coords)
                class_index = CLASS2IND[annotation["label"]]
                f_out.write(f"{class_index} {temp}\n")


process_data(train_filenames, train_labelnames, "train")
process_data(valid_filenames, valid_labelnames, "val")

# 테스트 데이터 처리
TEST_ROOT = "../data/test/DCM/"
test_pngs = {
    os.path.relpath(os.path.join(root, fname), start=TEST_ROOT)
    for root, _, files in os.walk(TEST_ROOT)
    for fname in files
    if fname.lower().endswith(".png")
}

os.makedirs(os.path.join(data_root_dir, "images", "test"), exist_ok=True)

for png in test_pngs:
    src_path = os.path.join(TEST_ROOT, png)
    dst_path = os.path.join(data_root_dir, "images", "test", os.path.basename(png))
    os.system(f"cp {src_path} {dst_path}")
