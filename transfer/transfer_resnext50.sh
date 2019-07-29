export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
myperson_root=/home/davinci/dnn/data/my_person/mxnet
rm -f nohup.out
nohup python transfer_learning_person.py \
  --data-dir ${myperson_root}/coco2017 \
  --test-dir ${myperson_root}/voc0712/train \
  --model ResNext50_32x4d --lr 0.001 --wd 0.0001 --classes 2 \
  --num-epochs 40 --batch-size 64 \
  --save-dir resnext50_32x4d_person_coco2017 \
  --lr-decay 0.75 \
  --num-gpus 1 --num-data-workers 8 &
