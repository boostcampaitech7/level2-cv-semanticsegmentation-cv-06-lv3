# BoostCamp AI Tech 7th CV-06 Semantic Segmentation


## 팀 구성 
---

<table>
  <tr>
    <td align="center"><a href="https://github.com/kkyungyoon"><img src="https://github.com/kkyungyoon.png" width="100px;" alt=""/><br /><sub><b>김경윤</b></sub></a><br /><a href="https://github.com/kkyungyoon" title="Code"></td>
    <td align="center"><a href="https://github.com/kimyoungseok3232"><img src="https://github.com/kimyoungseok3232.png" width="100px;" alt=""/><br /><sub><b>김영석</b></sub></a><br /><a href="https://github.com/kimyoungseok3232" title="Code"></td>
    <td align="center"><a href="https://github.com/Dangtae"><img src="https://github.com/Dangtae.png" width="100px;" alt=""/><br /><sub><b>신영태</b></sub></a><br /><a href="https://github.com/Dangtae" title="Code"></td>
    <td align="center"><a href="https://github.com/andantecode"><img src="https://github.com/andantecode.png" width="100px;" alt=""/><br /><sub><b>함로운</b></sub></a><br /><a href="https://github.com/andantecode" title="Code"></td>
     <td align="center"><a href="https://github.com/randfo42"><img src="https://github.com/randfo42.png" width="100px;" alt=""/><br /><sub><b>김태성</b></sub></a><br /><a href="https://github.com/randfo42" title="Code"></td>
	  <td align="center"><a href="https://github.com/taeyoung1005"><img src="https://github.com/taeyoung1005.png" width="100px;" alt=""/><br /><sub><b>박태영</b></sub></a><br /><a href="https://github.com/taeyoung1005" title="Code"></td>
  </tr>
</table>



## **소개**
---

24.11.11. ~ 24.11.28에 진행된 네부캠 7기 Hand Bone X-ray Image data Segmentation 대회입니다. 

Xray 이미지에서 손가락 뼈들을 Segmentation하는 Multi-Label  Task를 진행했습니다.

데이터 셋  : 2048 x 2048 크기의 손 뼈 X-ray 이미지. (Train : 800장, Test : 288장)

지표 : Dice Coefficient 

## **결과**
---

최종 순위 Public 5위, Private 6위 / 총 23 팀 


![image](https://github.com/user-attachments/assets/3e36d333-9ce3-4d94-8791-eeaf525005e0)



## **실험**
---

| 실험                       | <center>내용</center>                                                                                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **이미지 Resolution 고려**     | 높은 해상도를 가진 이미지이기에 Interpolation을 사용. 다만, 가장자리를 잘 포착하지 못해 이에 대한 성능 개선을 위한 실험. <br>보간 함수 변경, 원본 이미지 그대로 학습, Sliding Window, Super Resolution, SAM, Pixel Shuffling등을 적용하여 실험을 진행. |
| **Multi-Label Pixel 보완** | 손 등 뼈 부분의 픽셀이 Muti-Label인 경우, 모델 성능이 저조하여 이를 해결하기 위한 실험.<br>MixUp, Mask를 출력하고 Conv Layer로 다시 학습, 이미지 Crop 후 학습등 실험을 진행.                                                         |
| **후처리**                  | Mask 가장자리나, 비어있는 부분을 정제하기 위한 후처리.<br>OpenCV, Conv Layer, 클래스 별 Threshold 조정 등으로 실험을 진행.                                                                                         |
| **모델 구조 수정**             | 데이터에 맞는 모델 구조 수정 실험.<br>데이터의 해상도가 크기에 Batch를 못늘려 Batch Norm 변경, 학습 후 Conv Layer를 추가해서 Mask Fine Tuning, Auxilary Classifier 활용, Meta Data 활용, 배경 Pixel이 많은 걸 고려한 Loss 실험을 진행.   |
| **데이터 증강**               | 데이터에 맞는 증강을 찾기 위한 실험.<br>RandomScaleResized Crop, Rotate, Flip, Pseudo Labeling을 실험. <br>                                                                                       |
| **앙상블**                  | 다양한 모델을 합쳐 최대한의 성능을 이끌어 내기 위한 실험.<br>TTA, Label 별 Best Score Epoch 저장, Majority Voting, Weighted Majority Voting을 적용해봄.                                                         |

## **역할**
---

| 팀원     | <center>역할</center>                                                             |
| ------------ | ------------------------------------------------------------------------------- |
| **경윤** | MMSegmentation 모델 탐색, 해상도 변경 실험, 모델 수정을 통한 성능 개선, Loss 탐색, Pseudo Labeling, 후처리 |
| **영석** | 데이터 시각화 및 베이스라인에서 어그멘테이션, 모델 수정을 통해 성능개선 및 후처리                                  |
| **영태** | MMsegmentation 라이브러리 세팅 및 모델 탐색, 해상도 변경 실험, MixUP, TTA & Ensemble               |
| **로운** | SMP 모듈화, Sliding Window, 클래스별 threshold 적용 실험, multi-task learning              |
| **태성** | MMsegmentation 라이브러리 세팅. 해상도를 고려한 실험. Mask 정보를 재처리하는 실험                         |
| **태영** &nbsp; | 베이스라인 K-Fold, YOLO 실험, 앙상블 코드 수정, SR을 이용한 픽셀 보간, TTA                            |
