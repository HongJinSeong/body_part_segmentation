import copy
import platform
import random
from functools import partial

import numpy as np
import torch
import math
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader, DistributedSampler
import cv2
import mmcv

from mmseg.datasets import PIPELINES


########### mmdetection 의 Cutout을 수정하여 Random Erasing 으로 적용 
########### Cutout 적용시 생각보다 성능향상이 되지 않아서 cut한 영역에 random 값을 채워넣는 random erasing으로 적용
@PIPELINES.register_module()
class CutOut:
    """CutOut operation.
    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.
    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].

        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.

        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """

    def __init__(self,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0)):

        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:

            n_holes = (n_holes, n_holes)

        self.n_holes = n_holes

        self.fill_in = fill_in

        self.with_ratio = cutout_ratio is not None

        self.candidates = cutout_ratio if self.with_ratio else cutout_shape

        if not isinstance(self.candidates, list):

            self.candidates = [self.candidates]
	
    def __call__(self, results):

        """Call function to drop some regions of image."""

        h, w, c = results['img'].shape

        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
    
        for _ in range(n_holes):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            index = np.random.randint(0, len(self.candidates))

            if not self.with_ratio:

                cutout_w, cutout_h = self.candidates[index]

            else:

                cutout_w = int(self.candidates[index][0] * w)

                cutout_h = int(self.candidates[index][1] * h)

            x2 = np.clip(x1 + cutout_w, 0, w)

            y2 = np.clip(y1 + cutout_h, 0, h)
            
            #수정 ( 구멍뚫은 부분에 noise값으로 추가 )
            random_fill_in = np.random.randint(0, 255, (y2-y1,x2-x1,3)) 
            
            results['img'][y1:y2, x1:x2, :] = random_fill_in
            #기존 cutout 
            # results['img'][y1:y2, x1:x2, :] = self.fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in})'

        return repr_str
    

########## MMdetection 빼껴다 넣음############
########## detection과 다르게 segmentation은 label 이미지도 수정해야 되서 수정진행##############
@PIPELINES.register_module()
class RandomAffine:
    """Random affine transform data augmentation.
    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.
    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Default: 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Default: 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Default: (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Default: 2.
        border (tuple[int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Default: (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Default: (114, 114, 114).

    """

    def __init__(self,
                 max_rotate_degree=5.0,
                 max_translate_ratio=0.1,
                 scaling_ratio_range=(0.9, 1.1),
                 max_shear_degree=2.0,
                 border=(0, 0),
                 border_val=(0, 0, 0)):
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val

    def __call__(self, results):
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Center
        center_matrix = np.eye(3)
        center_matrix[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        center_matrix[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * width
        trans_y = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
            @ center_matrix)

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=(0, 0, 0))
        results['img'] = img
        results['img_shape'] = img.shape
        
        
        ####내가 임의로 수정한거 이상하면 수정필요#############
        ##mmdetection module 호출하여 수정한 것
        gt_img = results['gt_semantic_seg']
        gt_img =  cv2.warpPerspective(
                    gt_img,
                    warp_matrix,
                    dsize=(width, height),
                    borderValue=self.border_val) #translate로 생기는 영역 자체가 background 이기 때문에 background label 그대로 적용 
        
        #ground truth image affine transform 적용한 것 return 받도록 set
        results['gt_semantic_seg'] = gt_img
        
        # 원하는 형태로 이미지 나왔나 확인함 ( 이상없음 )
        # mmcv.imwrite(results['img'],'t.png')
        # mmcv.imwrite(results['gt_semantic_seg'],'tt.png')
        return results

  

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array([[np.cos(radian), -np.sin(radian), 0.],
                                    [np.sin(radian),
                                     np.cos(radian), 0.], [0., 0., 1.]])
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array([[scale_ratio, 0., 0.],
                                   [0., scale_ratio, 0.], [0., 0., 1.]])
        return scaling_matrix

    @staticmethod
    def _get_share_matrix(scale_ratio):
        scaling_matrix = np.array([[scale_ratio, 0., 0.],
                                   [0., scale_ratio, 0.], [0., 0., 1.]])
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees, y_shear_degrees):
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]])
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x, y):
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]])
        return translation_matrix
    
    
### 사람의 bodyparts는 붙어있기 때문에 flip되면 위치가 바뀌어야함
### pair로 두고 뒤집는 걸로 해결하기 
@PIPELINES.register_module()
class CustomRandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """
    def __init__(self, prob=None, direction='horizontal',flip_pair=None):
        self.prob = prob
        self.direction = direction
        self.flip_pair = flip_pair
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        th_val = random.random()
        flip = True if th_val < self.prob else False
        results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # mmcv.imwrite(results['img'],'ori.png')
            # mmcv.imwrite(results['gt_semantic_seg'],'ori_label.png')
            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction']).copy()
                # mmcv.imwrite(results['gt_semantic_seg'],'ori_label_flip.png')
                if self.flip_pair != None:
                    flip_gt = np.zeros_like(results[key])
                    for pair in self.flip_pair:
                        index1 = np.where(results[key]==pair[0])
                        flip_gt[index1] = pair[1]
                        
                        index2 = np.where(results[key]==pair[1])
                        flip_gt[index2] = pair[0]
                        
                        # print(pair[1])
                        # print(pair[0])
                        
                    index_body = np.where(results[key]==1)
                    flip_gt[index_body] = 1
                    
                    index_head = np.where(results[key]==14)
                    flip_gt[index_head] = 14
                        
                    results[key] = flip_gt
                        
            # mmcv.imwrite(results['img'],'t.png')
            # mmcv.imwrite(results['gt_semantic_seg'],'tt.png')
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'
