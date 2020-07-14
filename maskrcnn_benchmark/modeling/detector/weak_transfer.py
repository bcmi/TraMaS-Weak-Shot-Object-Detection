import torch
from torch import nn
import torch.nn.functional as F

# from maskrcnn_benchmark.structures.image_list import to_image_list
from ..backbone import build_backbone
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
#from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_predictors import make_roi_box_predictor
from maskrcnn_benchmark.modeling.roi_heads.box_head.inference import make_roi_box_post_processor
# from maskrcnn_benchmark.modeling.roi_heads.box_head.loss import make_roi_box_loss_evaluator

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target[None])

#         res = []
#         for k in topk:
#             correct_k = correct[:k].flatten().sum(dtype=torch.float32)
#             res.append(correct_k * (100.0 / batch_size))
#         return res

# # a, b: xyxy
# def iou(a, b):
#     A = (a[2] - a[0]) * (a[3] - a[1])
#     B = (b[2] - b[0]) * (b[3] - b[1])

#     lt = torch.max(a[:2], b[:2])
#     rb = torch.min(a[2:], b[2:])

#     #TO_REMOVE = 1
#     #wh = (rb - lt + TO_REMOVE).clamp(min=0)
#     wh = (rb - lt).clamp(min=0)
#     inter = wh[0] * wh[1]

#     return inter / (A + B - inter)

# # a: list of xyxy (4,N), b: xyxy (4,)
# def iou_4N(a, b):
#     A = (a[2] - a[0]) * (a[3] - a[1])
#     B = (b[2] - b[0]) * (b[3] - b[1])

#     lt = torch.max(a[:2,:], b[:2,None])
#     rb = torch.min(a[2:,:], b[2:,None])

#     #TO_REMOVE = 1
#     #wh = (rb - lt + TO_REMOVE).clamp(min=0)
#     wh = (rb - lt).clamp(min=0)
#     inter = wh[0] * wh[1]

#     return inter / (A + B - inter)

# # a: list of xyxy (4,N), b: list of xyxy (4,N)
# # return N x M
# def iou_4N_4N(a, b):
#     A = (a[2] - a[0]) * (a[3] - a[1])
#     B = (b[2] - b[0]) * (b[3] - b[1])

#     # lt: 2 x N x M
#     lt = torch.max(a[:2,:,None], b[:2,None])
#     rb = torch.min(a[2:,:,None], b[2:,None])

#     #TO_REMOVE = 1
#     #wh = (rb - lt + TO_REMOVE).clamp(min=0)
#     wh = (rb - lt).clamp(min=0)
#     inter = wh[0] * wh[1]

#     return inter / (A[:,None] + B[None] - inter)

# def iou_N4_N4(a, b):
#     A = (a[:,2] - a[:,0]) * (a[:,3] - a[:,1])
#     B = (b[:,2] - b[:,0]) * (b[:,3] - b[:,1])

#     # lt: 2 x N x M
#     lt = torch.max(a[:,None,:2], b[None,:,:2])
#     rb = torch.min(a[:,None,2:], b[None,:,2:])

#     #TO_REMOVE = 1
#     #wh = (rb - lt + TO_REMOVE).clamp(min=0)
#     #wh = (rb - lt).clamp(min=0)
#     wh = (rb - lt).clamp(min=0)
#     inter = wh[:,:,0] * wh[:,:,1]

#     return inter / (A[:,None] + B[None] - inter)

# def approx_max(x, dim=-1, beta=5.0, detach=True):
#     alpha = F.softmax(beta*x, dim)
#     if detach: alpha = alpha.detach()
#     return (x * alpha).sum(dim)

# def approx_max(x, dim=-1):
#     return torch.logsumexp(x, dim)


# class SingleConvRPNHead(nn.Module):
#     '''
#     This head supports two domains through num_classes, num_classes2.
#     '''
#     def __init__(self, cfg, in_channels, num_classes, num_classes2=0, has_bbox_pred=False, num_anchors=1):
#         super(SingleConvRPNHead, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels, in_channels, kernel_size=3, stride=1, padding=1
#         )
#         layers = [self.conv]
#         if num_classes:
#             self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)
#             layers.append(self.cls_logits)
#         if num_classes2:
#             self.cls_logits2 = nn.Conv2d(in_channels, num_classes2, kernel_size=1, stride=1)
#             layers.append(self.cls_logits2)
#         #self.num_anchors = num_anchors
#         assert num_anchors == 1
#         if has_bbox_pred:
#             self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
#             layers.append(self.bbox_pred)
#         self.has_bbox_pred = has_bbox_pred

#         for l in layers:
#             torch.nn.init.normal_(l.weight, std=0.003)
#             torch.nn.init.constant_(l.bias, 0)
#         # self.detach_box = cfg.WEAK.DETACH_BOX

#     def forward(self, x, domain=1):
#         cls_logits, cls_logits2 = [], []
#         bbox_reg = []
#         for feature in x:
#             t = F.relu(self.conv(feature))
#             if domain & 1:
#                 cls_logits.append(self.cls_logits(t))
#             if domain & 2:
#                 cls_logits2.append(self.cls_logits2(t))
#             if self.has_bbox_pred:
#                 # if self.detach_box: t = t.detach()
#                 bbox_reg.append(self.bbox_pred(t))
#         return cls_logits, cls_logits2, bbox_reg


class RPNModule(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()
        # self.cfg = cfg.clone()

        # head = SingleConvRPNHead(
        #     cfg, in_channels, 
        #     num_classes=cfg.WEAK.NUM_CLASSES,
        #     num_classes2=0,
        #     has_bbox_pred=False,
        #     num_anchors=1 # anchor_generator.num_anchors_per_location()[0]
        # )
        # self.head = head
        # print ('RPN num classes:', cfg.WEAK.NUM_CLASSES)

    def forward(self, images, features, det, targets=None):
        assert len(features) == 1, "only support 1 level for now"

        with torch.no_grad():
            det_features = det.backbone(images.tensors)
            # det_proposals, _, det_objectness, det_rpn_box_regression, det_anchors = det.rpn(
            #     images, det_features, targets=None, return_more=True)
            det_proposals, _ = det.rpn(images, det_features, targets=None)
            x, det_results, _ = det.roi_heads(det_features, det_proposals, targets=None)

            # self.det_proposals, self.det_objectness, self.det_rpn_box_regression, self.det_anchors, self.det_results = \
            #     det_proposals, det_objectness, det_rpn_box_regression, det_anchors, det_results

        proposals = []
        for d in det_results:
            if len(d) == 0:
                # unluckily no proposal, put a default box
                # print (d)
                # d.bbox = torch.as_tensor([[0,0,d.size[0],d.size[1]]], dtype=d.bbox.dtype, device=d.bbox.device)
                d.bbox = torch.as_tensor([[d.size[0]*0.1,d.size[1]*0.1,d.size[0]*0.8,d.size[1]*0.8]],
                                         dtype=d.bbox.dtype, device=d.bbox.device)
                d.add_field('scores', torch.as_tensor([0.05], dtype=d.bbox.dtype, device=d.bbox.device))
            # make a deep copy of the proposals
            b = BoxList(d.bbox, d.size, d.mode)
            b.extra_fields['scores'] = d.get_field('scores')
            proposals.append(b)
        return proposals, {}


class FastRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels, has_bbox_pred=False):
        super(FastRCNNPredictor, self).__init__()
        num_inputs = in_channels

        if not "WithoutPool" in cfg.MODEL.ROI_BOX_HEAD.PREDICTOR:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        # +1 for background
        self.cls_score = nn.Linear(num_inputs, cfg.WEAK.NUM_CLASSES + 1)
        weights = [self.cls_score.weight]
        biases = [self.cls_score.bias]
        # self.cls_score2 = nn.Linear(num_inputs, cfg.WEAK.NUM_CLASSES2 + 1)
        # weights += [self.cls_score2.weight]
        # biases += [self.cls_score2.bias]

        # self.OICR = cfg.WEAK.OICR
        # if self.OICR:
        #     self.oicr_cls = nn.Linear(num_inputs, cfg.WEAK.OICR * (cfg.WEAK.NUM_CLASSES + 1))
        #     weights += [self.oicr_cls.weight]
        #     biases += [self.oicr_cls.bias]

        self.bilinear = cfg.WEAK.BILINEAR
        if cfg.WEAK.BILINEAR:
            self.det_score = nn.Linear(num_inputs, cfg.WEAK.NUM_CLASSES)
            weights += [self.det_score.weight]
            biases += [self.det_score.bias]
            # self.det_score2 = nn.Linear(num_inputs, cfg.WEAK.NUM_CLASSES2)
            # weights += [self.det_score2.weight]
            # biases += [self.det_score2.bias]

        if has_bbox_pred:
            num_bbox_reg_classes = 1
            self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
            weights += [self.bbox_pred.weight]
            biases += [self.bbox_pred.bias]

        for w in weights:
            nn.init.normal_(w, mean=0, std=0.001)
        for b in biases:
            nn.init.constant_(b, 0)

    def forward(self, x, domain=None):
        if hasattr(self,"avgpool"):
            x = self.avgpool(x)
            x = x.view(x.size(0), x.size(1))
        # bbox_pred = self.bbox_pred(x)
        bbox_pred = None
        cls_logit = self.cls_score(x)
        # if self.OICR:
        #     oicr_cls = self.oicr_cls(x)
        # else:
        oicr_cls = None
        if self.bilinear:
            det_logit = self.det_score(x)
            return cls_logit, det_logit, bbox_pred, oicr_cls
        return cls_logit, bbox_pred, oicr_cls


class ROIBoxHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = FastRCNNPredictor(cfg, self.feature_extractor.out_channels)
        # assert cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
        self.post_processor = make_roi_box_post_processor(cfg)
        # self.loss_evaluator = make_roi_box_loss_evaluator(cfg, allow_low_quality_matches=True)
        self.WEAK = cfg.WEAK

    def compute_obj(self, features, proposals):
        x = self.feature_extractor(features, proposals)
        cls_logits, det_logits, box_regression, oicr_cls_logits = self.predictor(x)

        cls_prob = F.softmax(cls_logits, 1)  # R x (C+1)
        det_prob = det_logits.sigmoid()  # R x C, range [0,1]

        num_boxes = [len(_) for _ in proposals]

        scores = (cls_prob[:, 1:] * det_prob).max(1)[0]
        scores = scores.detach().split(num_boxes)

        return scores

    def forward(self, features, proposals, targets=None, domain=None, disable_regression=False):
        x = self.feature_extractor(features, proposals)
        assert self.WEAK.BILINEAR

        cls_logits, det_logits, box_regression, oicr_cls_logits = self.predictor(x, domain)

        cls_prob = F.softmax(cls_logits, 1)  # R x (C+1)
        det_prob = det_logits.sigmoid()  # R x C, range [0,1]

        self.cls_prob, self.det_prob = cls_prob, det_prob

        num_boxes = [len(_) for _ in proposals]
        list_det_prob = (det_prob * self.WEAK.ROI_BETA).split(num_boxes)
        list_det_prob = [F.softmax(_, 0) for _ in list_det_prob]  # R x C

        # ignore background class (index 0)
        list_cls_prob = cls_prob[:, 1:].split(num_boxes)

        # final score is the product of the two
        list_cls_prob = [c * d for c,d in zip(list_cls_prob, list_det_prob)]
        list_orgscores = [p.get_field("scores") for p in proposals]

        # list_orgscores = [p / p.max() if len(p) else p for p in list_orgscores]

        # K = self.WEAK.OICR
        # if K > 0:
        #     oicr_cls_prob = F.softmax(oicr_cls_logits.view(-1, K, self.WEAK.NUM_CLASSES + 1), 2)

        self.list_det_prob, self.list_cls_prob, self.list_orgscores = list_det_prob, list_cls_prob, list_orgscores
        img_cls_prob = torch.stack([_.sum(0) for _ in list_cls_prob])

        if not self.training:
            # testing mode
            scores = cls_prob[:, 1:] * det_prob
            # if K > 0:
            #     scores = (scores + oicr_cls_prob[:, :, 1:].sum(1)) / (K + 1)
            scores = scores.detach().split(num_boxes)

            for i,p in enumerate(proposals):
                if len(scores[i]) > 0:
                    s = scores[i]*self.WEAK.SCORE_COEF + list_orgscores[i][:,None]*((1-self.WEAK.SCORE_COEF) / list_orgscores[i].max())
                    sm,lm = s.max(1)
                    s = s * 0
                    s[torch.arange(len(s), dtype=torch.long), lm] = sm
                else:
                    s = scores[i]

                proposals[i].add_field("scores", torch.cat((s[:,:1]*0, s), dim=1).view(-1))
                proposals[i].bbox = proposals[i].bbox.repeat(1, self.WEAK.NUM_CLASSES+1).view(-1, 4)

                if not self.post_processor.bbox_aug_enabled:
                    proposals[i] = self.post_processor.filter_results(proposals[i], self.WEAK.NUM_CLASSES+1)

            return proposals, None

        # training mode
        img_labels = torch.stack([t.get_field("img_labels") for t in targets])
        # another (probably better) way to avoid numerical instability is to do
        # torch's logsumexp or logsoftmax, then do binary_cross_entropy_with_logits.
        loss_cls = F.binary_cross_entropy(img_cls_prob.clamp(1e-10, 1-1e-10), img_labels)

        det_reduce = det_prob.max(1)[0]
        orgscores = torch.cat(list_orgscores)
        loss_obj = F.mse_loss(det_reduce, orgscores.detach()) * self.WEAK.OBJ_WEIGHT

        losses = {"roi_cls": loss_cls, "roi_obj": loss_obj}

        # OICR. not used
        # if K > 0:
        #     nimg = len(targets)
        #     # nimg x [regions x 21]
        #     list_oicr_cls_prob = oicr_cls_prob.split(num_boxes)
        #     device = oicr_cls_prob.device

        #     # list_scores = [a*self.WEAK.SCORE_COEF + b*(1-self.WEAK.SCORE_COEF) for a,b in zip(list_cls_prob, list_orgscores)]
        #     list_scores = [a*b[:,None].detach() for a,b in zip(list_cls_prob, list_orgscores)]

        #     y = [torch.zeros((num_boxes[i], K), dtype=torch.long, device=device) for i in range(nimg)]
        #     w = [torch.zeros((num_boxes[i], K), device=device) for i in range(nimg)]
        #     for i in range(nimg):
        #         if num_boxes[i] == 0: continue
        #         for c in range(self.WEAK.NUM_CLASSES):
        #             if img_labels[i, c]:
        #                 last_prob = list_scores[i][:, c]
        #                 # last_prob = list_cls_prob[i][:, c]
        #                 # if len(last_prob) == 0: continue
        #                 for k in range(K):
        #                     jc = last_prob.argmax()
        #                     # y[k, i, jc] = c
        #                     ious = iou_N4_N4(proposals[i].bbox[jc:jc+1], proposals[i].bbox)
        #                     mask = ious[0] > 0.5
        #                     y[i][mask, k] = c + 1
        #                     w[i][:, k] = last_prob[jc]
        #                     last_prob = list_oicr_cls_prob[i].detach()[:, k, c + 1]
        #     y = torch.cat(y).view(-1)
        #     w = torch.cat(w).view(-1)  #.detach()
        #     loss_oicr = torch.mean(F.cross_entropy(oicr_cls_logits, y, reduction='none') * w)
        #     losses['oicr'] = loss_oicr

        return proposals, losses


class WeakTransfer(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = RPNModule(cfg, self.backbone.out_channels)
        self.roi_head = ROIBoxHead(cfg, self.backbone.out_channels)

        self.WEAK = cfg.WEAK
        assert self.WEAK.MODE == "transfer" or self.WEAK.MODE == "extract"

        # torch.autograd.set_detect_anomaly(True)
        from .generalized_rcnn import GeneralizedRCNN
        from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer, Checkpointer
        cfg2 = cfg.clone()
        cfg2.merge_from_file(cfg.WEAK.CFG2)
        cfg2.defrost()
        # print ("cfg.MODEL.ROI_HEADS.NMS", cfg2.MODEL.ROI_HEADS.NMS, cfg.MODEL.ROI_HEADS.NMS, '\n')
        cfg2.MODEL.ROI_HEADS.NMS = cfg.MODEL.ROI_HEADS.NMS
        if hasattr(cfg.MODEL.ROI_HEADS, "SCORE_THRESH_CFG2"):
            cfg2.MODEL.ROI_HEADS.SCORE_THRESH = cfg.MODEL.ROI_HEADS.SCORE_THRESH_CFG2
        # cfg2.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
        # cfg2.MODEL.WEAK_DET = ""
        cfg2.freeze()
        # wrap in a container to avoid being counted by model_serialization / state_dict()
        class container: pass
        self.det_container = container()
        self.det_container.model = GeneralizedRCNN(cfg2)
        output_dir = cfg.WEAK.CFG2[:cfg.WEAK.CFG2.rfind('/')]  #cfg.WEAK.CFG2_OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg2, self.det_container.model, save_dir=output_dir)
        checkpointer.load("fail")
        self.det_container.model.eval()

    '''
    An optimization of the pipeline to avoid separate OCUD and MIL training would be
    sending in data from both domains simultaneously, e.g.,
        def forward(self, images1, targets1=None, images2=None, targets2=None)
    But this complicates implementation.
    '''
    def forward(self, images1, targets1=None):
        losses = {}
        rpn_feat1 = self.backbone(images1.tensors)

        if self.training:
            eye = torch.eye(self.WEAK.NUM_CLASSES, dtype=rpn_feat1[-1].dtype, device=rpn_feat1[-1].device)
            for t in targets1:
                t.add_field("img_labels", eye[t.get_field("labels") - 1, :].sum(0).clamp_(0,1))

        self.det_container.model.to(images1.tensors.device)

        proposals1, rpn_losses = self.rpn(images1, rpn_feat1, det=self.det_container.model, targets=targets1)
        losses.update(rpn_losses)

        boxes1, roi_losses = self.roi_head(rpn_feat1, proposals1, targets1)
        if not self.training: return boxes1

        losses.update(roi_losses)
        return losses

    def forward_backbone(self, images):
        with torch.no_grad():
            self.rpn_feat = self.backbone(images.tensors)
            self.det_container.model.to(images.tensors.device)
            self.rpn.det_features = self.det_container.model.backbone(images.tensors)

    def compute_obj(self, targets, images=None):
        if images is not None:
            self.forward_backbone(images)
        with torch.no_grad():
            det_obj = self.det_container.model.roi_heads.box.compute_obj(self.rpn.det_features, targets)
            weak_obj = self.roi_head.compute_obj(self.rpn_feat, targets)
            objs = [a*self.WEAK.SCORE_COEF + b*((1-self.WEAK.SCORE_COEF)/b.max()) for a,b in zip(weak_obj, det_obj)]
        return objs
