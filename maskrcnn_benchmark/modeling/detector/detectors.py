# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .weak_transfer import WeakTransfer

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, "WeakTransfer": WeakTransfer}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
