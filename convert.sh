#!/bin/bash

echo "Extract Train data Segmentation Mask"
python convert.py ./ --data_type train --save_root ./train_mask

echo "Extract Val data Segmentation Mask"
python convert.py ./ --data_type val --save_root ./val_mask
