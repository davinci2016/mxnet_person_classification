#!/usr/bin/env bash
mx_root=~/src/dnn/mxnet/my_person/

python demo_person.py --model resnet50_v2 \
  --load-params  ${mx_root}/finetune/resnet50_v2_param_coco2017/0.9312-myperson-resnet50_v2-5-best.params
##  --load-params  ${mx_root}/train/params_resnet50_v2_person/myperson-resnet50_v2-19.params
