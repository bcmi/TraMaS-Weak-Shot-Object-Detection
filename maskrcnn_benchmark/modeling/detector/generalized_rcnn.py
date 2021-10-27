# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


# def toimgclass(targets):
#     img_classes=torch.zeros((len(targets),80))
#     for i,t in enumerate(targets):
#         classes=t.get_field('labels_img').long()
#         for c in classes:
#             img_classes[i][c]=1
#     return img_classes.cuda()
def get_target_label(targets):
    return torch.cat([t.get_field('labels') for t in targets])

def get_img_labels(targets):
    img_labels=torch.zeros((len(targets),80))
    for i,t in enumerate(targets):
        img_labels[i]=t.get_field("img_labels")
    return img_labels.cuda()

def focal_loss(x, p = 1, c = 0.1):
    return torch.pow(1 - x, p) * torch.log(c + x)

def weight_init_(models):
        for m in models.modules():
            if isinstance(m,nn.Conv2d):
                print(1)
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight.data,1)
                nn.init.constant_(m.bias.data,0)

class MaskGenerator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.last_conv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                           nn.BatchNorm2d(256), nn.ReLU(),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                           nn.BatchNorm2d(256), nn.ReLU(),
                                           nn.Conv2d(256, 80, kernel_size=1, stride=1),
                                           )
        self.creterion=nn.MultiLabelSoftMarginLoss()

        weight_init_(self.last_conv)
    def forward(self,features,labels=None):

        x = self.last_conv(features)

        # constant BG scores
        bg = torch.ones_like(x[:, :1])
        x = torch.cat([bg, x], 1)

        bs, c, h, w = x.size()

        masks = F.softmax(x, dim=1)

        # reshaping
        features = x.view(bs, c, -1)
        masks_ = masks.view(bs, c, -1)

        # classification loss
        cls_1 = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1))

        # focal penalty loss
        cls_2 = focal_loss(masks_.mean(-1), \
                            p=3, \
                            c=0.01)

        # adding the losses together
        cls = cls_1[:, 1:] + cls_2[:, 1:]

        # foreground stats
        masks_ = masks_[:, 1:]

        if labels is None:
            return None,masks

        cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

        loss_cls=self.creterion(cls,labels)
        mask_loss={'mask_loss':(loss_cls.mean()+cls_fg.mean())*0.1}
        return mask_loss,masks

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.mask_generator=MaskGenerator()
        self.conv_fusion=nn.Sequential(nn.Conv2d(81+1024,1024,kernel_size=1),
                                        nn.BatchNorm2d(1024),nn.ReLU())
        self.extract_features=cfg.EXTRACT_FEATURES
        self.feature_extractor = make_roi_box_feature_extractor(cfg, 1024)
        weight_init_(self.conv_fusion)



    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        #print(targets[0].get_field("weights"))
        if self.training:
            img_labels=get_img_labels(targets)
        else:
            img_labels=None

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        pre_features,features = self.backbone(images.tensors)
        mask_loss,masks=self.mask_generator(pre_features,img_labels)
        masks=F.interpolate(masks,features.shape[2:])
        features=[self.conv_fusion(torch.cat((features,masks),1))]



        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        
        if self.extract_features:
            roi_features=self.feature_extractor(features,targets)
            return F.adaptive_avg_pool2d(roi_features, (1,1)),get_target_label(targets)

        if self.training:
            losses = {}
            losses.update(mask_loss)
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
