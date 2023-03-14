# body_part_segmentation

AI 히어로즈 대회

첫 semantic segmentation 대회 

Segformer 라는 transformer 기반 semantic segmentation 모델 fine tune으로 진행 
Augmentation에 집중하여 진행(최종 적용 : resize / random crop / photometricdistortion / custom random flip / random affine / cutout)

MMsegmentation이라는 tool 사용 (mmseg/mmdet 등등 자주 사용하게 될거 같음)

이미지를 flip 시킬 경우 좌우반전이 되는 부분을 늦게 신경써줘서 의미있는 경향성 찾는 충분한 시도 못해봄..

최종 6등
