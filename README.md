# TransMaS
This repository is the official pytorch implementation of the following paper:
> **[NIPS2021 Mixed Supervised Object Detection by TransferringMask Prior and Semantic Similarity]**
> 
> Yan Liu<sup>∗</sup>,  Zhijie Zhang<sup>∗</sup>,  Li Niu<sup>†</sup>,  Junjie Chen,  Liqing Zhang<sup>†</sup>
> 
> MoE Key Lab of Artificial, IntelligenceDepartment of Computer Science and Engineering, Shanghai Jiao Tong University

## Setup
Follow the instructions in [Installation](https://github.com/mikuhatsune/wsod_transfer/blob/master/INSTALL.md) to build the projects.

## Data
Follow instructions in [README.old.md](https://github.com/mikuhatsune/wsod_transfer/blob/master/README.old.md) to setup COCO and VOC datasets folder and place the coco and voc files under folder `./datasets`. Annotations for the COCO-60, and VOC datasets on [Google Drive](https://drive.google.com/drive/folders/1HhCGksyo1Eza7LhQtISelvyRHNL1iohc?usp=sharing)
- `coco60_train2017_21987.json`, `coco60_val2017_969.json` : place under folder `./datasets/coco/annotations/`
- `voc_2007_trainval.json`, `voc_2007_test.json`: place under `./datasets/voc/VOC2007/`

## Checkpoints
we provide the model checkpoints of object detection network and mil classifier. All checkpoint files are on [Google Drive](https://drive.google.com/drive/folders/1HhCGksyo1Eza7LhQtISelvyRHNL1iohc?usp=sharing), place the files under folder `./output/coco60_to_voc/`

## Evaluation
The test results of **Ours<sup>*</sup>(single-scale)** on VOC2007 tsetset in [the main paper]() can be obtained by Executing the following commands:
```
python -m torch.distributed.launch --nproc_per_node=2 tools/test_net.py --config-file wsod/coco60_to_voc/mil_it0.yaml OUTPUT_DIR "output/coco60_to_voc/mil_it2" MODEL.WEIGHT "output/coco60_to_voc/mil_it2/model_final.pth" WEAK.CFG2 "output/coco60_to_voc/odn_it2/config.yml"
```
## Acknowledgements
Thanks to [WSOD with Progressive Knowledge Transfer](https://github.com/mikuhatsune/wsod_transfer) providing the base architecture, iterative training strategy and data annotations for our project, we further proposed to transfer mask prior and semantic similarity to bridge the gap between novel categories and base categorie and added the code for Mask Generator and SimNet .

