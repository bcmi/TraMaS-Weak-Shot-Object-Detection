# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .mil_classifier import MILCls

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, "MILCls": MILCls}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
