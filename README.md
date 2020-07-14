# WSOD with Progressive Knowledge Transfer

The code accompanies the following paper:

* Boosting Weakly Supervised Object Detection with Progressive Knowledge Transfer. Yuanyi Zhong, Jianfeng Wang, Jian Peng, Lei Zhang. ECCV 2020.

Relevant diff from the original maskrcnn-benchmark in commit [ecc6b5f](https://github.com/mikuhatsune/wsod_transfer/commit/ecc6b5f82f67a4293fe7b201aabfcc759626a82b) .
Please follow the instructions of maskrcnn-benchmark [`README.old.md`](README.old.md) to setup the environment.

## Key files
- [`wsod/coco60_to_voc`](wsod/coco60_to_voc): YAML configs for the COCO60-to-VOC experiment.
- [`wsod/pseudo_label.py`](wsod/pseudo_label.py): pseudo ground truth mining script on source and target datasets.
- [`maskrcnn_benchmark/modelling/detector/weak_transfer.py`](maskrcnn_benchmark/modelling/detector/weak_transfer.py): the multi-instance learning in the paper.

## Data

Annotations for the `COCO-60`, `COCO-60-full` and `VOC` datasets on [Google Drive](https://drive.google.com/drive/folders/1SrDVRttw6K6xSBJFwu0JnFU6YnEJQaDN?usp=sharing).

- `coco60_train2017_21987.json`, `coco60_val2017_969.json`, `coco60full_train2017_118287.json`, `coco60full_val2017_5000.json`: place under folder `./datasets/coco/annotations/`
- `voc_2007_trainval.json`, `voc_2007_test.json`: place under `./datasets/voc/VOC2007/`

## Demo to reproduce the COCO-60 to VOC experiment

Run the following commands to train 3 iterations of the algorithm described in the paper. (They can be wrapped in a single shell script to run together.)

##### Initial Iteration (K=0)

```bash
# result: output/coco60_to_voc/ocud_it0
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file wsod/coco60_to_voc/ocud_it0.yaml
# result: output/coco60_to_voc/mil_it0
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file wsod/coco60_to_voc/mil_it0.yaml
# result: datasets/voc/VOC2007/voc_2007_trainval_coco60-to-voc_it0_0.8.json
# and datasets/coco/annotations/coco60_{train,val}2017_coco60-to-voc_it0_0.8
python wsod/pseudo_label.py output/coco60_to_voc/mil_it0 coco60_train2017_21987 coco60_val2017_969 coco60-to-voc 0 0.8 | tee output/coco60_to_voc/mil_it0/pseudo.txt
```

##### 1st Refinement (K=1)

```bash
# ocud_it1
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file wsod/coco60_to_voc/ocud_it1.yaml --start_iter 0
# mil_it1
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file wsod/coco60_to_voc/mil_it1.yaml --start_iter 0
# pseudo GT it1
python wsod/pseudo_label.py output/coco60_to_voc/mil_it1 coco60_train2017_21987 coco60_val2017_969 coco60-to-voc 1 0.8 | tee output/coco60_to_voc/mil_it1/pseudo.txt
```

##### 2nd Refinement (K=2)

We can make duplicates `ocud_it2.yaml` and `mil_it2.yaml` for this, or reuse the previous configs and specify the paths as follows.

```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file wsod/coco60_to_voc/ocud_it1.yaml --start_iter 0 OUTPUT_DIR "output/coco60_to_voc/ocud_it2" MODEL.WEIGHT "output/coco60_to_voc/ocud_it1/model_final.pth" DATASETS.TRAIN "('coco60_train2017_coco60-to-voc_it1_0.8','coco60_val2017_coco60-to-voc_it1_0.8','voc_2007_trainval_coco60-to-voc_it1_0.8_cocostyle')"

python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file wsod/coco60_to_voc/mil_it1.yaml --start_iter 0 OUTPUT_DIR "output/coco60_to_voc/mil_it2" MODEL.WEIGHT "output/coco60_to_voc/mil_it1/model_final.pth" WEAK.CFG2 "output/coco60_to_voc/ocud_it2/config.yml"

python wsod/pseudo_label.py output/coco60_to_voc/mil_it2 coco60_train2017_21987 coco60_val2017_969 coco60-to-voc 2 0.8 | tee output/coco60_to_voc/mil_it2/pseudo.txt
```

##### (Optional) Distill a Faster RCNN

This retrains a Faster RCNN (R50C4) from the pseudo GT mined in step K=2.

```
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file wsod/coco60_to_voc/distill_resnet50c4.yaml
```

### Demo Result

Here are the VOC2007 test APs of the Demo above. Note that we report the mAP@IoU=0.5 under the VOC07 11-point metric in our paper, which is a bit lower than the area under PR curve.

`mil_it2`:

```
use_07_metric=True:
mAP: 0.5875
aeroplane       : 0.5851
bicycle         : 0.4720
bird            : 0.6876
boat            : 0.4561
bottle          : 0.4812
bus             : 0.7835
car             : 0.7515
cat             : 0.8028
chair           : 0.2962
cow             : 0.8010
diningtable     : 0.1465
dog             : 0.7994
horse           : 0.7006
motorbike       : 0.6749
person          : 0.5640
pottedplant     : 0.1211
sheep           : 0.6998
sofa            : 0.5831
train           : 0.7261
tvmonitor       : 0.6180
```

Due to randomness (and this code being a refactored version..), the numbers may vary from run to run and slightly differ from the paper's. But the difference should be rather limited. The example result here gives 58.75% mAP at K=2 which is higher than that in the paper.

Demo outputs are on [Google Drive](https://drive.google.com/drive/folders/1SrDVRttw6K6xSBJFwu0JnFU6YnEJQaDN?usp=sharing).
