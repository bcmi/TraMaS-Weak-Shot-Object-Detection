from tqdm import tqdm
import os, sys
import json
import torch, numpy as np
from maskrcnn_benchmark.data.build import build_dataset
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.config import cfg
paths_catalog = import_file(
    "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
)
DatasetCatalog = paths_catalog.DatasetCatalog

local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
if local_rank != 0:
    sys.exit()

folder, coco_train, coco_val, tag, it, th = sys.argv[1:]
print ('pseudo label params:', folder, coco_train, coco_val, tag, it, th)

###############################################################################
# VOC 2007 trainval pseudo labeling

# output label file names
json_voc = 'voc_2007_trainval_%s_it%s_%s.json' % (tag, it, th)

path = folder + '/inference/voc_2007_trainval/predictions.pth'
print ("read", path)
d_trainval = torch.load(path)

dataset_list = ("voc_2007_trainval",)
transforms = None
datasets = build_dataset(dataset_list, transforms, DatasetCatalog, False)

gt_trainval = [datasets[0].get_groundtruth(idx) for idx in range(len(datasets[0]))]

p_trainval = {}
p_trainval.update(zip(datasets[0].ids, zip(gt_trainval, d_trainval)))


with open('datasets/voc/VOC2007/voc_2007_trainval.json','r') as f:
    d = json.load(f)

th = float(th)
annos = []
id = 0
for img_id, (t, p) in tqdm(p_trainval.items(), mininterval=20):
    img_labels = set(t.get_field('labels').tolist())
    p = p.resize(t.size)

    boxes = p.bbox.cpu().numpy()
    scores = p.get_field('scores').tolist()
    labels = p.get_field('labels').tolist()
    sortidx = np.argsort(scores)[::-1]
    labels_hit = set()
    for i in sortidx:
        l = labels[i]
        if l in img_labels and (scores[i] > th or l not in labels_hit):
            labels_hit.add(l)
            bbox = boxes[i].copy()
            bbox[2:] -= bbox[:2] - 1
            bbox = bbox.tolist()
            id += 1
            anno = {'area': bbox[2]*bbox[3], 'iscrowd': 0, 'image_id': int(img_id), 
                    'bbox': bbox, 'category_id': l, 'id': id, 'ignore': 0}
            annos.append(anno)

print ('threshold', th, 'result #instances', len(annos))
d['annotations'] = annos

fn = 'datasets/voc/VOC2007/%s' % json_voc
print ("save to " + fn + '\n')
with open(fn,'w') as f:
    json.dump(d, f)


###############################################################################
# COCO-60 (COCO 2017 train, val) pseudo labeling

# json_coco = '%s_trainval_it%s_%s.json' % (tag, it, th)

path = folder + '/inference/' + coco_train + '/predictions.pth'
print ("read", path)
d_train = torch.load(path)

path = folder + '/inference/' + coco_val + '/predictions.pth'
print ("read", path)
d_val = torch.load(path)


dataset_list = (coco_train, coco_val)
transforms = None
datasets = build_dataset(dataset_list, transforms, DatasetCatalog, False)

gt_train = [datasets[0].get_groundtruth(idx) for idx in range(len(datasets[0]))]
gt_val = [datasets[1].get_groundtruth(idx) for idx in range(len(datasets[1]))]

# p_trainval = {}
# p_trainval.update(zip(datasets[0].ids, zip(gt_train, d_train)))
# p_trainval.update(zip(datasets[1].ids, zip(gt_val, d_val)))
p_train = dict(zip(datasets[0].ids, zip(gt_train, d_train)))
p_val = dict(zip(datasets[1].ids, zip(gt_val, d_val)))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from tqdm import tqdm

def boxlist_overlap1(boxlist1, boxlist2):
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # intersection over area1
    overlap = inter / area1[:, None]
    return overlap
    # Alternative: intersection over union
    # iou = inter / (area1[:, None] + area2 - inter)
    # return iou

# ov_th: overlap threshold
# score_th: score threshold
def mine_boxes(p_trainval, ov_th, score_th, mined_class_label=1, visualize=False):
    mined_images = 0
    mined_boxes = 0
    # lines = []
    # has_mined = []
    annos = []
    id = 0

    for img_id, (t, p) in tqdm(p_trainval.items(), mininterval=20):
    # for img_id, (t, p) in p_trainval.items():
        p = p.resize(t.size)
        p.add_field('labels', (p.get_field('labels') > 0).to(torch.long) * mined_class_label)
        p = boxlist_nms(p, 0.4)
        s = p.get_field('scores')
        # Strategy 1: keep at least one box per image even the score is low
        # p = p[s >= min(score_th, s.max())]
        # Strategy 2: keep on high score ones
        p = p[s >= score_th]

        if len(p) and len(t):
            #ious = boxlist_iou(p, anno)
            ious = boxlist_overlap1(p, t)
            # try:
            ious = ious.max(1)[0]
            # except:
            #     print (p,t,ious)
            p = p[ious < ov_th]

        if len(p):
            mined_images += 1
            mined_boxes += len(p)

            # pn = [{'class': '_mined_', 'rect': p.bbox[i].tolist()} for i in range(len(p))]
            # l[3] = l[3] + pn
            # has_mined.append(True)
            del p.extra_fields['scores']
            t = cat_boxlist((t, p))
            # print (t.bbox, t.get_field('labels'))

            if visualize:
                #img = d.get_img(img_id)
                path = datasets[0].coco.loadImgs(img_id)[0]['file_name']
                img = Image.open(os.path.join(datasets[0].root, path)).convert('RGB')
                plt.imshow(img)
                for i in range(len(t)):
                    x0, y0, x1, y1 = t.bbox[i]
                    w, h = x1-x0+1, y1-y0+1
                    plt.gca().add_patch(Rectangle((x0, y0), w, h, alpha=0.9,
                                                  facecolor='none', edgecolor='green', linewidth=1.5))
                for i in range(len(p)):
                    x0, y0, x1, y1 = p.bbox[i]
                    w, h = x1-x0+1, y1-y0+1
                    plt.gca().add_patch(Rectangle((x0, y0), w, h, alpha=0.9,
                                                  facecolor='none', edgecolor='red', linewidth=1))
                #plt.title(str(d.lines[id]))
                print (img_id, t, p, mined_images)
                plt.show()
        # else:
        #     has_mined.append(False)

        # lines.append(l)
        boxes = t.bbox.cpu().numpy().copy()
        labels = t.get_field('labels').tolist()
        for i in range(len(boxes)):
            bbox = boxes[i] #.copy()
            bbox[2:] -= bbox[:2] - 1
            bbox = bbox.tolist()
            id += 1
            anno = {'area': bbox[2]*bbox[3], 'iscrowd': 0, 'image_id': int(img_id), 
                    'bbox': bbox, 'category_id': labels[i], 'id': id, 'ignore': 0}
            annos.append(anno)

    print ('mined_images', mined_images, 'mined_boxes', mined_boxes)
    return annos

with open('datasets/coco/annotations/%s.json' % coco_train,'r') as f:
    d1 = json.load(f)
with open('datasets/coco/annotations/%s.json' % coco_val,'r') as f:
    d2 = json.load(f)

# d = d2.copy()
# d['images'] = d1['images'] + d2['images']

for split, d, p in zip(['train','val'], [d1, d2], [p_train, p_val]):
    mined_class_label = 1
    d['categories'].append({'supercategory': 'none', 'id': mined_class_label, 'name': '_mined'})

    annos = mine_boxes(p, ov_th=0.1, score_th=th, mined_class_label=mined_class_label)

    print ('threshold', th, 'result #instances', len(annos))
    d['annotations'] = annos

    json_coco = 'coco60_%s2017_%s_it%s_%s.json' % (split, tag, it, th)
    fn = 'datasets/coco/annotations/%s' % json_coco
    print ("save to " + fn + '\n')
    with open(fn,'w') as f:
        json.dump(d, f)
