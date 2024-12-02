import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

csv = pd.read_csv('./output/best_score.csv') ## 후처리 적용할 csv파일 경로

def fill_inner_holes(mask):
    """
    외곽은 유지하면서 안쪽 빈 공간만 채우는 함수.
    """
    # 외곽을 반전하여 외부를 찾음
    h, w = mask.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)  # OpenCV 요구 사항으로 가장자리 추가
    inverted_mask = cv2.bitwise_not(mask)  # 0과 1을 반전
    # Flood Fill로 외부 채우기
    cv2.floodFill(inverted_mask, flood_fill_mask, (0, 0), 1)

    # 내부 빈 공간 채우기
    filled_mask = ((inverted_mask) > 1).astype(np.uint8)

    return filled_mask

def largest_connected_component(mask):
    """
    채워진 마스크에서 가장 큰 연결 성분만 남기기.
    """
    # 연결 성분 분석
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 가장 큰 연결 성분 선택 (배경 제외)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype(np.uint8)
    return largest_mask

def rle_to_mask(rle, shape):
    """ RLE 데이터를 디코딩하여 2D 마스크 배열 생성 """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    rle_pairs = [int(x) for x in rle.split()]
    for idx in range(0, len(rle_pairs), 2):
        start_pixel = rle_pairs[idx]
        run_length = rle_pairs[idx + 1]
        mask[start_pixel:start_pixel + run_length] = 1
    return mask.reshape(shape)

def mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def find_closest_points(poly1, poly2, left_side=True):
    """poly2에서 가장 왼쪽 또는 오른쪽에 있는 점과 poly1에서 가장 가까운 점을 찾습니다."""
    if left_side: target_point_index = np.argmin(poly2[:, 0])  # x 좌표 기준으로 최소값 인덱스
    else: target_point_index = np.argmax(poly2[:, 0])  # x 좌표 기준으로 최대값 인덱스
    
    target_point = poly2[target_point_index]
    
    # poly1에서 가장 가까운 점을 찾기
    min_dist = float('inf')
    closest_pair = None
    
    for i, pt1 in enumerate(poly1):
        dist = np.linalg.norm(pt1 - target_point)
        if dist < min_dist:
            min_dist = dist
            closest_pair = (i, target_point_index)

    return closest_pair

def merge_polygons(poly1, poly2):
    """
    두 폴리곤을 가장 가까운 점 기준으로 병합.
    각 폴리곤의 점 순서를 재정렬하여 자연스럽게 이어지도록 함.
    """
    poly1 = poly1.squeeze()  # (N, 1, 2) -> (N, 2)
    poly2 = poly2.squeeze()

    # 가장 가까운 두 점 찾기
    leftidx1, leftidx2 = find_closest_points(poly1, poly2, left_side=True)

    poly1 = np.roll(poly1, -leftidx1, axis=0)
    poly2 = np.roll(poly2, -leftidx2, axis=0)

    rightidx1, rightidx2 = find_closest_points(poly1, poly2, left_side=False)

    poly1 = poly1[:rightidx1+1]
    poly2 = np.flip(poly2[:rightidx2+1], axis=0)

    # 합쳐서 하나의 폴리곤으로 만듬
    merged_polygon = np.concatenate([poly1, poly2])

    return merged_polygon

def merge_two_bigcomponent(mask):
    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 폴리곤으로 변환 (점 좌표 가져오기)
    polygons = [contour.reshape(-1, 2) for contour in contours]

    if len(polygons) > 1:
        polygons = [merge_polygons(polygons[0], polygons[1])]
    else: return mask
    
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(new_mask, pts=[np.int32(p) for p in polygons], color=1)

    return new_mask

def post(csv):
    shape = (2048,2048)
    res = []

    csv = csv.fillna('')
    fxdel = 0
    fxadd = 0
    for i in csv.itertuples():
        if not i.rle:
            #print(i.image_name, i._2, i.Index//29, 0)
            res.append({'image_name': i.image_name,
                        'class': i._2,
                        'rle': i.rle
                        })
            continue
        mask = rle_to_mask(i.rle, shape)
        final_mask = fill_inner_holes(mask)
        if i._2 != 'finger-3': final_mask = largest_connected_component(final_mask)
        else: final_mask = fill_inner_holes(merge_two_bigcomponent(final_mask) | final_mask)
        final_rle = mask_to_rle(final_mask)
        
        res.append({'image_name': i.image_name,
                    'class': i._2,
                    'rle': final_rle
                    })
        if mask.sum() != final_mask.sum():
            diff = int(mask.sum())-int(final_mask.sum())
            print(i.image_name, i._2, i.Index//29, diff)
            if diff < 0: fxadd -= diff
            else: fxdel += diff
    print(f'deleted fixel: {fxdel:<5}, added fixel: {fxadd:<5}, fixed fixel: {fxdel+fxadd:<5}')

post(csv)