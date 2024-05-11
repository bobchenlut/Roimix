# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
import matplotlib.pyplot as plt
from mmcv.visualization import imshow_bboxes,imshow_det_bboxes
import torch.nn.functional as F
from mmdet.core.bbox import bbox_overlaps
from mmcv.ops import nms

@DETECTORS.register_module()
class ROIMix(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(ROIMix, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.init_assigner_sampler()

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        with torch.no_grad():
            img,gt_bboxes,gt_labels=self.roimix(img,
                   img_metas,
                   gt_bboxes,
                   gt_labels,
                   gt_bboxes_ignore=None)

        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    @torch.no_grad()
    def roimix(self,img,
               img_metas,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               beta=0.1,
               alpha=0.1,):
        x = self.extract_feat(img)
        proposal_list = self.rpn_head.simple_test_rpn(x,img_metas)

        for i in range(len(proposal_list)):
           proposal_list[i],_=nms(proposal_list[i][:,:4].contiguous(),proposal_list[i][:,-1].contiguous(),iou_threshold=0.1)

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        bboxes_list=[e.pos_bboxes for e in sampling_results]
        label_list=[e.pos_gt_labels for e in sampling_results]
        labels=torch.cat(label_list,dim=0)
        rois=bbox2roi(bboxes_list)
        permuation=torch.randperm(rois.size(0))
        rois_permuated=rois[permuation]

        beta_dist=torch.distributions.Beta(torch.tensor(alpha),torch.tensor(beta))
        lamda=beta_dist.sample((1,))
        lamda_=max(lamda,1-lamda).to(img.device)


        for roi,roi_,label,k in zip(rois,rois_permuated,labels,range(len(rois))):
            b1=roi[0].int()
            img_single=img[b1]
            x1,y1,x2,y2=roi[1:].int()
            patch=img_single[:,y1:y2,x1:x2]

            b2 = roi_[0].int()
            img_single_ = img[b2]
            x1_, y1_, x2_, y2_ = roi_[1:].int()

            patch_ = img_single_[:, y1_:y2_, x1_:x2_]
            patch_size=patch.shape[1:]
            patch_=F.interpolate(patch_.unsqueeze(0),patch_size).squeeze(0)

            dist_patch=patch*lamda_+(1-lamda_)*patch_

            id = min(int(k * num_imgs / len(roi)), num_imgs - 1)
            img[id,:,y1:y2,x1:x2]=dist_patch

            ious=bbox_overlaps(gt_bboxes[id],roi[1:].reshape(1,4))
            is_gt=torch.any(ious>0.8)
            if not is_gt:
                gt_bboxes[id]=torch.cat((gt_bboxes[id],roi[1:].reshape(1,4)),0)
                gt_labels[id]=torch.cat((gt_labels[id],label.reshape(1,)),0)
        # class_names=("holothurian","echinus","scallop","starfish")
        # imshow_det_bboxes(img[0].permute(1,2,0).to("cpu").numpy(),gt_bboxes[0].to("cpu").numpy(),
        #                   gt_labels[0].to("cpu").numpy(),class_names=class_names,win_name="fig1",
        #                   text_color='red')
        # imshow_det_bboxes(img[1].permute(1, 2, 0).to("cpu").numpy(),gt_bboxes[1].to("cpu").numpy(),
        #                   gt_labels[1].to("cpu").numpy(),class_names=class_names,win_name='fig2',
        #                   text_color='red')
        # imshow_bboxes(img[0].permute(1, 2, 0).to("cpu").numpy(), gt_bboxes[0].to("cpu").numpy())
        # imshow_bboxes(img[1].permute(1, 2, 0).to("cpu").numpy(), gt_bboxes[1].to("cpu").numpy())

        return img,gt_bboxes,gt_labels



