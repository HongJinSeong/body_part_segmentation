#!/bin/bash

echo "Download Densepose coco json"
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_minival.json

echo "Download COCO 2014 train images"
wget http://images.cocodataset.org/zips/train2014.zip

echo "Download COCO 2014 val images"
wget http://images.cocodataset.org/zips/val2014.zip

echo "unzip train images"
unzip train2014.zip

echo "unzip val images"
unzip val2014.zip