# TransMaS
This repository is the official pytorch implementation of the following paper:
> **[NIPS2021 Mixed Supervised Object Detection by TransferringMask Prior and Semantic Similarity](https://arxiv.org/pdf/2110.14191.pdf)**
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
We provide the model checkpoints of object detection network and MIL classifier. All checkpoint files are on [Google Drive](https://drive.google.com/drive/folders/1HhCGksyo1Eza7LhQtISelvyRHNL1iohc?usp=sharing), place the files under folder `./output/coco60_to_voc/`

## Evaluation
The test results of **Ours<sup>*</sup>(single-scale)** on VOC2007 test set in [the main paper](https://arxiv.org/pdf/2110.14191.pdf) can be reproduced by executing the following commands:
```
python -m torch.distributed.launch --nproc_per_node=2 tools/test_net.py --config-file wsod/coco60_to_voc/mil_it0.yaml OUTPUT_DIR "output/coco60_to_voc/mil_it2" MODEL.WEIGHT "output/coco60_to_voc/mil_it2/model_final.pth" WEAK.CFG2 "output/coco60_to_voc/odn_it2/config.yml"
```

## Resources
We have summarized the existing papers and codes on weak-shot learning in the following repository:
[https://github.com/bcmi/Awesome-Weak-Shot-Learning](https://github.com/bcmi/Awesome-Weak-Shot-Learning)

## Acknowledgements
Thanks to [WSOD with Progressive Knowledge Transfer](https://github.com/mikuhatsune/wsod_transfer) providing the base architecture, iterative training strategy, and data annotations for our project. We further transfer mask prior and semantic similarity to bridge the gap between novel categories and base categorie by adding the code for Mask Generator and SimNet.

