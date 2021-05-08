#! /bin/bash

 python tools/train.py configs/hrnet/faster_rcnn_hrnetv2p_w18_dynamic.py

for i in {40..80}
do

python tools/test.py configs/hrnet/faster_rcnn_hrnetv2p_w18_dynamic.py work_dirs/DHRNet/hrnetw18_MinIoURandomCrop_2backbone_dynamicFromS4_lr0002_sig22_fanin/epoch_$i.pth --out results/DHRNet/hrnetw18_MinIoURandomCrop_2backbone_dynamicFromS4_lr0002_sig22_fanin/e$i.pkl --eval bbox

done

