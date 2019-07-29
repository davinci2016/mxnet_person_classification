#!/usr/bin/env bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
myperson_root=/home/davinci/dnn/data/my_person/mxnet
#nohup 
python person.py \
  --data-dir ${myperson_root}/coco2017 \
  --model ResNext50_32x4d --lr 0.01 --wd 0.001 --classes 2 \
  --num-epochs 60 --batch-size 64 \
  --save-dir ResNext50_32x4d_person_coco2017
#&
