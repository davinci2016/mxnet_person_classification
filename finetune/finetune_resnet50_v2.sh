#!/usr/bin/env bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
myperson_root=/home/davinci/dnn/data/my_person/mxnet
#nohup
python person.py \
  --data-dir ${myperson_root}/coco2017 \
  --model resnet50_v2 --lr 0.01 --wd 0.001 --classes 2 \
  --num-epochs 60 --batch-size 1 \
  --save-dir resnet50_v2_person_coco2017-2
#&