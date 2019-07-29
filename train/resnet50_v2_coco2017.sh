#!/usr/bin/env bash
myperson_root=/home/davinci/dnn/data/my_person/mxnet

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

nohup python train_person.py \
  --data-dir ${myperson_root}/coco2017 \
  --model resnet50_v2 --mode hybrid \
  --lr 0.001 --wd 0.0001 \
  --lr-mode step --num-epochs 60 --batch-size 64 --num-gpus 1 -j 8 \
  --warmup-epochs 0 \
  --save-dir params_resnet50_v2_person_coco2017 \
  --imagenet-mode 0 \
  --classes 2 \
  --num-training-samples 118293 \
  --logging-file train_resnet50_v2_coco2017.log \
  --save-mxnet-dir mxnet_models &

  #--lr 0.001 --wd 0.0001 \
  #--use-rec \
  #--rec-train ${rec_path}/train.rec --rec-train-idx ${rec_path}/train.idx \
  #--rec-val ${rec_path}/val.rec --rec-val-idx ${rec_path}/val.idx \
  #--use-pretrained \
