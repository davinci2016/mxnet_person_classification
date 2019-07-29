#!/usr/bin/env bash
mx_root=~/src/dnn/mxnet/my_person/

python demo_person.py --model resnet50_v2 \
  --saved-params  ${mx_root}/finetune/resnet50_v2_param/0.9343-myperson-resnet50_v2-6-best.params \
  --input-pic /home/davinci/dnn/data/my_person/mxnet/coco2017/val/1/000000507893.jpg
