import os
import cv2
import pathlib
import argparse
import multiprocessing


import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util

from random import randint
from tqdm import tqdm
from pycocotools.coco import COCO


def get_config():
    parser = argparse.ArgumentParser(description='Extract Segmentation Mask.')

    parser.add_argument('coco_folder', type=pathlib.Path)
    parser.add_argument('--data_type', type=str, choices=['train', 'val'],
                        help='select whether the data you want to convert is train or val.')
    parser.add_argument("--save_root", type=pathlib.Path)

    args = parser.parse_args()

    return args


def get_annotation_filename(data_type: str) -> str:
    
    if data_type == "train":
        json_name = 'datasets/densepose_coco_2014_train.json'

    elif data_type == "val":
        json_name = 'datasets/densepose_coco_2014_minival.json'
    
    else:
        raise ValueError('you must input "train" or "val".')

    return json_name


def GetDensePoseMask(Polys):
    MaskGen = np.zeros([256,256])
    for i in range(1,15):
        if(Polys[i-1]):
            current_mask = mask_util.decode(Polys[i-1])
            MaskGen[current_mask>0] = i
    return MaskGen


def make_mask(coco: COCO, img_ids: list, img_folder_name: str) -> None:

    for img_id in tqdm(img_ids):

        # Load the image
        im = coco.loadImgs(img_id)[0]  

        # Load Anns for the selected image.
        ann_ids = coco.getAnnIds( imgIds=im['id'] )
        anns = coco.loadAnns(ann_ids)

        # Now read and b
        img_name = args.coco_folder / img_folder_name / im['file_name']
        cur_image=cv2.imread(str(img_name))

        # I_vis = cur_image.copy()
        result_mask = np.zeros(cur_image.shape[:2])

        for ann in anns:  
            bbr =  np.array(ann['bbox']).astype(int) # the box.
            if( 'dp_masks' in ann.keys()): # If we have densepose annotation for this ann, 
                bbox_mask = GetDensePoseMask(ann['dp_masks'])

                x1,y1,x2,y2 = bbr[0],bbr[1],bbr[0]+bbr[2],bbr[1]+bbr[3]
                x2 = min( [ x2, cur_image.shape[1] ] );  y2 = min( [ y2, cur_image.shape[0] ] )

                mask_img = cv2.resize( bbox_mask, (int(x2-x1),int(y2-y1)) ,interpolation=cv2.INTER_NEAREST)
                print(np.unique(mask_img))
                # MaskBool = np.tile((mask_img==0)[:,:,np.newaxis],[1,1,3])

                result_mask[y1:y2,x1:x2] = mask_img
        break

        save_name = pathlib.Path(im['file_name'])
        save_name = save_name.stem + '.png'
        save_path = args.save_root / save_name
        cv2.imwrite(str(save_path), result_mask)


if __name__=="__main__":

    args = get_config()
    args.save_root.mkdir(exist_ok=True)

    json_filename = get_annotation_filename(args.data_type)
    coco = COCO(args.coco_folder / json_filename) 

    # Get img id's for the minival dataset.
    im_ids = coco.getImgIds()

    img_folder_name = args.data_type + "2014"
    make_mask(coco, im_ids, img_folder_name)
    


#  A  A
# (‘ㅅ‘=)
# J.M.Seo
# From Alchera Inc.