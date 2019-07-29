export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
myperson_root=/home/davinci/dnn/data/my_person/mxnet
mx_root=~/src/dnn/mxnet/my_person/

python transfer_learning_person.py \
  --data-dir ${myperson_root}/coco2017 \
  --test-dir ${myperson_root}/coco2014/train \
  --model ResNext50_32x4d --lr 0.001 --wd 0.0001 --classes 2 \
  --num-epochs 40 --batch-size 64 \
  --save-dir resnext50_32x4d_person_coco2017 \
  --lr-decay 0.75 \
  --num-gpus 1 --num-data-workers 8 \
  --work-mode 1 \
  --load-params ${mx_root}/transfer/resnext50_32x4d_person_coco2017-infotm/0.9346-myperson-ResNext50_32x4d-6-best.params


## --load-params ${mx_root}/transfer/resnext50_32x4d_person_coco2017/0.9384-myperson-ResNext50_32x4d-5-best.params
## --test-dir ${myperson_root}/voc0712/train \ ##0.920
## --test-dir ${myperson_root}/coco2014/train \ ##0.955



## --load-params ${mx_root}/transfer/resnext50_32x4d_person_coco2017-infotm/0.9346-myperson-ResNext50_32x4d-6-best.params
## VOC0712: Test-acc: 0.926
## COC2014: Test-acc: 0.957