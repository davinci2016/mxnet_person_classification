#!/usr/bin/env bash
mx_root=~/src/dnn/mxnet/my_person/

python demo_person.py --model ResNext50_32x4d \
  --load-params  ${mx_root}/transfer/resnext50_32x4d_person_coco2017-infotm/0.9346-myperson-ResNext50_32x4d-6-best.params
