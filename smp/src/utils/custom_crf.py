import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from skimage.color import gray2rgb
import numpy as np

class DenseCRF:
    def __init__(self, shape):
        """
        DenseCRF 초기화 (이미지 크기 shape)
        :param shape: (height, width) 또는 (height, width, channel)
        """
        self.shape = shape

    def set_parameters(self, num_classes=2, sxy=3, srgb=10):
        """
        CRF 모델의 파라미터 설정
        :param num_classes: 클래스 수 (여기서는 2)
        :param sxy: 공간 제약 정도 (기본 3)
        :param srgb: 색상 제약 정도 (기본 10)
        """
        self.num_classes = num_classes
        self.sxy = sxy
        self.srgb = srgb
        self.dcrf = dcrf.DenseCRF2D(self.shape[1], self.shape[0], self.num_classes)  # (width, height)와 클래스 수

    def crf(original_image, mask_img):
    
        # Converting annotated image to RGB if it is Gray scale
        if(len(mask_img.shape)<3):
            mask_img = gray2rgb(mask_img)

        # Converting the annotations RGB color to single 32 bit integer
        annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
        
        # Convert the 32bit integer color to 0,1, 2, ... labels.
        colors, labels = np.unique(annotated_label, return_inverse=True)

        n_labels = 2
        
        #Setting up the CRF model
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
            
        #Run Inference for 10 steps 
        Q = d.inference(10)

        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)

        return MAP.reshape((original_image.shape[0], original_image.shape[1]))
