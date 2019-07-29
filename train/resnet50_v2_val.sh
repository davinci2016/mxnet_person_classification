#!/usr/bin/env bash
imagenet_root=/home/davinci/.mxnet/datasets/imagenet
rec_path=${imagenet_root}/rec

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

python train_person.py \
  --model resnet50_v2 --mode hybrid \
  --lr 0.001 --lr-mode cosine --num-epochs 10 --batch-size 64 --num-gpus 1 -j 8 \
  --warmup-epochs 5 \
  --save-dir params_resnet50_v2 \
  --use-pretrained \
  --imagenet-mode 2 \
  --save-mxnet-dir mxnet_models \
  --data-dir /home/davinci/dnn/data/my_person/mxnet/voc0712
  #--use-rec \
  #--rec-train ${rec_path}/train.rec --rec-train-idx ${rec_path}/train.idx \
  #--rec-val ${rec_path}/val.rec --rec-val-idx ${rec_path}/val.idx \