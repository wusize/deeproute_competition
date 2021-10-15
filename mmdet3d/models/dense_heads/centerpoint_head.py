import copy
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from mmcv.ops import RoIAlignRotated
from torch import nn

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply
from det3d.models import builder as external_builder

@HEADS.register_module()
class SeparateHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SeparateHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@HEADS.register_module()
class DCNSeparateHead(BaseModule):
    r"""DCNSeparateHead for CenterHead.

    .. code-block:: none
            /-----> DCN for heatmap task -----> heatmap task.
    feature
            \-----> DCN for regression tasks -----> regression tasks

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        dcn_config (dict): Config of dcn layer.
        num_cls (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 num_cls,
                 heads,
                 dcn_config,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(DCNSeparateHead, self).__init__(init_cfg=init_cfg)
        if 'heatmap' in heads:
            heads.pop('heatmap')
        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = build_conv_layer(dcn_config)

        self.feature_adapt_reg = build_conv_layer(dcn_config)

        # heatmap prediction head
        cls_head = [
            ConvModule(
                in_channels,
                head_conv,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                bias=bias,
                norm_cfg=norm_cfg),
            build_conv_layer(
                conv_cfg,
                head_conv,
                num_cls,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)
        ]
        self.cls_head = nn.Sequential(*cls_head)
        self.init_bias = init_bias
        # other regression target
        self.task_head = SeparateHead(
            in_channels,
            heads,
            head_conv=head_conv,
            final_kernel=final_kernel,
            bias=bias)
        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        self.cls_head[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for DCNSepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['heatmap'] = cls_score

        return ret


@HEADS.register_module()
class CenterHead(BaseModule):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: None.
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        loss_cls (dict): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_heatmap_convs (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 roi_align=None,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(CenterHead, self).__init__(init_cfg=init_cfg)

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(builder.build_head(separate_head))

        if roi_align is not None:
            # print(roi_align, flush=True)
            self.roi_align = RoIAlignRotated(**roi_align)
        else:
            self.roi_align = None

    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        # print(x.shape, flush=True)
        ret_dicts = []

        x = self.shared_conv(x)
        # print(x.shape, flush=True)
        for task in self.task_heads:
            task_out = task(x)
            # print(task_out.keys(), flush=True)
            # exit()
            ret_dicts.append(task_out)

        return ret_dicts

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_single, feats)

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
        # transpose heatmaps, because the dimension of tensors in each task is
        # different, we have to use numpy instead of torch to do the transpose.
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 8),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    #vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        #vx.unsqueeze(0),
                        #vy.unsqueeze(0)
                    ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, img_metas, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],),
                 #preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the \
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the \
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the \
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # Apply NMS in birdeye view

            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_gpu(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg['nms_thr'],
                    pre_maxsize=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])

                print(len(top_scores), len(selected), flush=True)

            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def get_single_batch_sample_points(self, box, num_point=5):
        """
        Args:
            box dict: Decoded bbox, scores and labels after nms.

        Returns:
            points
        """
        top_center = box['bbox'].corners[:, [1, 2, 5, 6]].mean(1)
        # print(torch.tan(box['bbox'].tensor[0, -1]), flush=True)
        if num_point == 1:
            points = top_center.view(-1, 1, 3)

        elif num_point == 5:
            points = top_center.new_zeros(top_center.shape[0], 5, 3)

            back_middle = box['bbox'].corners[:, [1, 2]].mean(1)
            left_middle = box['bbox'].corners[:, [1, 5]].mean(1)
            right_middle = box['bbox'].corners[:, [2, 6]].mean(1)
            front_middle = box['bbox'].corners[:, [5, 6]].mean(1)

            points[:, 0] = top_center
            points[:, 1] = front_middle
            points[:, 2] = back_middle
            points[:, 3] = left_middle
            points[:, 4] = right_middle
            # vector = top_center[0] - left_middle[0]
            # print(vector[0] / vector[1], flush=True)
            # exit()
        else:
            raise NotImplementedError()
        box['sample_points'] = points

        return points

    def get_single_bev_features(self, box, pts_feat):
        """
        Args:
            box dict: Decoded bboxs, scores, labels and sample_points
            pts_feat tensor: (1, 512, 200, 200)
        Returns:
            box dict: Decoded bboxs, scores, labels, sample_points and bev_features
        """

        if pts_feat.dim() == 3:
            pts_feat = pts_feat[None]
        pc_range = self.train_cfg['point_cloud_range']
        voxel_size = self.train_cfg['voxel_size']
        out_size_factor = self.train_cfg['out_size_factor']

        sample_points_2d = box['sample_points'][:, :, :2]   # num_objs, 5, 2
        num_objs, num_samples, _ = sample_points_2d.shape
        sample_points_2d[:, :, 0] = (
                                            sample_points_2d[:, :, 0] - pc_range[0]
                                    ) / voxel_size[0] / out_size_factor
        sample_points_2d[:, :, 1] = (
                                            sample_points_2d[:, :, 1] - pc_range[1]
                                    ) / voxel_size[1] / out_size_factor
        # sample_points_2d = sample_points_2d.flip([-1]).view(1, 1, -1, 2)
        sample_points_2d = sample_points_2d.view(1, 1, -1, 2)

        #
        # x_0 = sample_points_2d[0, 0, 0, 0].long().item()
        # y_0 = sample_points_2d[0, 0, 0, 1].long().item()
        # x_1 = (sample_points_2d[0, 0, 0, 0] + 0.5).long().item()
        # y_1 = (sample_points_2d[0, 0, 0, 1] + 0.5).long().item()
        _, h, w, _ = pts_feat.shape
        sample_points_2d[:, :, 0] = sample_points_2d[:, :, 0] * 2 / (w - 1) - 1.0
        sample_points_2d[:, :, 1] = sample_points_2d[:, :, 1] * 2 / (h - 1) - 1.0
        # print(x_0, x_1, y_0, y_1, flush=True)

        # pts_feat[0, 0, y_0, x_0] = 9999.0
        # pts_feat[0, 0, y_0, x_1] = 9999.0
        # pts_feat[0, 0, y_1, x_0] = 9999.0
        # pts_feat[0, 0, y_1, x_1] = 9999.0

        sampled_features = F.grid_sample(pts_feat, sample_points_2d,
                                         align_corners=True)    # 1, C, 1, num_objs*num_samples
        # print(sample_points_2d[0, 0, 0], sampled_features[0, 0, 0, 0], flush=True)
        #
        # print(pts_feat[0, 0, y_0, x_0], pts_feat[0, 0, y_0, x_1],
        #       pts_feat[0, 0, y_1, x_0], pts_feat[0, 0, y_1, x_1], flush=True)
        #
        #
        # exit()
        sampled_features = sampled_features[0, :, 0].T.view(num_objs,
                                                            num_samples, -1)
        box['bev_features'] = sampled_features

        return sampled_features

    def get_single_roi_aligned_features(self, box, pts_feat):
        """
        Args:
            box dict: Decoded bboxs, scores, labels and sample_points
            pts_feat tensor: (1, 512, 200, 200)
        Returns:
            box dict: Decoded bboxs, scores, labels, sample_points and bev_features
        """
        if pts_feat.dim() == 3:
            pts_feat = pts_feat[None]
        pc_range = self.train_cfg['point_cloud_range']
        voxel_size = self.train_cfg['voxel_size']
        out_size_factor = self.train_cfg['out_size_factor']

        bev = box['bbox'].bev  # num_objs, 5 x, y, w, h, yaw

        num_objs = bev.shape[0]

        rois = bev.new_zeros(num_objs, 6)

        rois[:, 1] = (
                             bev[:, 0] - pc_range[0]
                     ) / voxel_size[0] / out_size_factor

        rois[:, 2] = (
                             bev[:, 1] - pc_range[1]
                     ) / voxel_size[1] / out_size_factor

        rois[:, 3] = (
                         bev[:, 2]
                     ) / voxel_size[0] / out_size_factor

        rois[:, 4] = (
                         bev[:, 3]
                     ) / voxel_size[1] / out_size_factor

        rois[:, 5] = bev[:, 4]


        roi_ailgned_features = self.roi_align(pts_feat, rois)

        roi_ailgned_features = roi_ailgned_features.view(num_objs,
                                                         pts_feat.shape[1],
                                                         -1)
        # print(rois[0], flush=True)
        # print(roi_ailgned_features[0, 0], flush=True)

        roi_ailgned_features = roi_ailgned_features.permute(0, 2, 1)

        box['bev_features'] = roi_ailgned_features

        return roi_ailgned_features


    def prepare_batched_rois(self, boxes):
        batch_size = len(boxes)
        box_length = boxes[0]['bbox'].tensor.shape[1]
        feature_vector_length = boxes[0]['bev_features'].shape[2] \
                                * boxes[0]['bev_features'].shape[1]
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg'] \
            if self.training else self.test_cfg['max_per_img']

        rois = boxes[0]['bbox'].tensor.new_zeros((batch_size,
                                                  max_objs, box_length))
        roi_scores = boxes[0]['scores'].new_zeros((batch_size, max_objs))
        roi_labels = boxes[0]['labels'].new_zeros((batch_size, max_objs),
                                                       dtype=torch.long)
        roi_features = boxes[0]['bev_features'].new_zeros((batch_size,
                                                           max_objs,
                                                           feature_vector_length))
        example = {}
        for i in range(batch_size):
            num_objs =  boxes[i]['bev_features'].shape[0]
            num_objs = min(num_objs, max_objs)
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target

            box_preds = boxes[i]['bbox'].tensor

            if self.bbox_coder.code_size == 9:
                # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            rois[i, :num_objs] = box_preds[:num_objs]
            roi_labels[i, :num_objs] = boxes[i]['labels'][:num_objs] + 1
            roi_scores[i, :num_objs] = boxes[i]['scores'][:num_objs]
            roi_features[i, :num_objs] = boxes[i]['bev_features'][:num_objs
                                         ].contiguous().view(num_objs, -1)

        example['rois'] = rois
        example['roi_labels'] = roi_labels
        example['roi_scores'] = roi_scores
        example['roi_features'] = roi_features

        example['has_class_labels'] = True

        return example


@HEADS.register_module()
class TwoStageCenterHead(BaseModule):
    """TwoStageCenterHead for CenterPoint.
    Args:
    """

    def __init__(self,
                 first_stage_cfg,
                 roi_head_cfg,
                 num_points=5,
                 train_cfg=None,
                 test_cfg=None,
                 loss_weights=[0, 1.0],
                 freeze=True,
                 end2end=False,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(TwoStageCenterHead, self).__init__(init_cfg=init_cfg)
        if end2end:
            assert not freeze

        self.first_stage = builder.build_head(first_stage_cfg)
        self.roi_head = external_builder.build_roi_head(roi_head_cfg)
        self.roi_head_cfg = roi_head_cfg
        self.end2end = end2end
        self.freeze = freeze
        self.num_points = num_points
        self.loss_weights=loss_weights

    def forward(self, pts_feats):
        if self.freeze:
            with torch.no_grad():
                outs = self.first_stage(pts_feats)
        else:
            outs = self.first_stage(pts_feats)
        for task_outs in outs:
            for level_task_outs, level_pts_feats in zip(task_outs, pts_feats):
                level_task_outs['bev_feature'] = level_pts_feats

        return outs

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        first_stage_bboxs = self.first_stage.get_bboxes(preds_dicts, img_metas)
        for batch_idx in range(len(first_stage_bboxs)):
            tmp = {'bbox': first_stage_bboxs[batch_idx][0], 'scores': first_stage_bboxs[batch_idx][1],
                   'labels': first_stage_bboxs[batch_idx][2]}
            first_stage_bboxs[batch_idx] = tmp
        bev_features = torch.cat([pred_dict['bev_feature']
                                  for pred_dict in preds_dicts[0]], dim=1)
        if self.first_stage.roi_align is None:
            _ = multi_apply(self.first_stage.get_single_batch_sample_points,
                            first_stage_bboxs,
                            [self.num_points for _ in first_stage_bboxs])
            _ = multi_apply(self.first_stage.get_single_bev_features,
                            first_stage_bboxs,
                            list(bev_features))
        else:
            _ = multi_apply(self.first_stage.get_single_roi_aligned_features,
                            first_stage_bboxs,
                            list(bev_features))

        exit()

        batched_rois = self.first_stage.prepare_batched_rois(first_stage_bboxs)

        return self.forward_second_stage(batched_rois, img_metas)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, img_metas, **kwargs):
        first_stage_losses = self.first_stage.loss(gt_bboxes_3d, gt_labels_3d,
                                                   preds_dicts, img_metas)
        for k in first_stage_losses.keys():
            first_stage_losses[k] = first_stage_losses[k] * self.loss_weights[0]
        first_stage_bboxs = self.first_stage.get_bboxes(preds_dicts, img_metas)
        for batch_idx in range(len(first_stage_bboxs)):
            tmp = {'bbox': first_stage_bboxs[batch_idx][0], 'scores': first_stage_bboxs[batch_idx][1],
                   'labels': first_stage_bboxs[batch_idx][2]}
            first_stage_bboxs[batch_idx] = tmp

        bev_features = torch.cat([pred_dict['bev_feature']
                                  for pred_dict in preds_dicts[0]], dim=1)
        if self.first_stage.roi_align is None:
            _ = multi_apply(self.first_stage.get_single_batch_sample_points,
                            first_stage_bboxs,
                            [self.num_points for _ in first_stage_bboxs])
            _ = multi_apply(self.first_stage.get_single_bev_features,
                            first_stage_bboxs,
                            list(bev_features))
        else:
            _ = multi_apply(self.first_stage.get_single_roi_aligned_features,
                            first_stage_bboxs,
                            list(bev_features))


        batched_rois = self.first_stage.prepare_batched_rois(first_stage_bboxs)
        batched_rois = self.add_gt_boxes(batched_rois,
                                         gt_bboxes_3d, gt_labels_3d)

        second_stage_losses = self.forward_second_stage(batched_rois, img_metas)
        # print(first_stage_losses, second_stage_losses, flush=True)
        for k in second_stage_losses.keys():
            second_stage_losses[k] = second_stage_losses[k] * self.loss_weights[1]

        second_stage_losses.update(first_stage_losses)

        return second_stage_losses

    def add_gt_boxes(self, batched_rois, gt_bboxes_3d, gt_labels_3d):
        batch_size, num_objs, box_length = batched_rois['rois'].shape
        gt_boxes_and_cls = batched_rois['rois'].new_zeros(batch_size, num_objs, box_length + 1)

        for batch_idx in range(batch_size):
            valid_num = min(num_objs, len(gt_bboxes_3d[batch_idx]))
            rot = gt_bboxes_3d[batch_idx].tensor[:valid_num, 6]
            rot_sin = torch.sin(rot)
            rot_cos = torch.cos(rot)
            rot_aligned = torch.atan2(rot_sin, rot_cos)
            gt_boxes_and_cls[batch_idx, :valid_num, :box_length] \
                = gt_bboxes_3d[batch_idx].tensor[:valid_num]
            gt_boxes_and_cls[batch_idx, :valid_num, 6] = rot_aligned
            gt_boxes_and_cls[batch_idx, :valid_num, -1] \
                = gt_labels_3d[batch_idx][:valid_num].float() + 1.0

        batched_rois['gt_boxes_and_cls'] = gt_boxes_and_cls

        return batched_rois

    def forward_second_stage(self, batched_rois, img_metas=None):
        if not self.end2end:
            for k, v in batched_rois.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if v.requires_grad:
                    batched_rois[k] = v.detach()

        batch_dict = self.roi_head(batched_rois, training=self.training)

        if self.training:
            loss_dict = self.roi_head.loss()

            return loss_dict
        else:
            return self._get_bboxes(batch_dict, img_metas)

    def _get_bboxes(self, batch_dict, img_metas):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts list[dict]: Prediction results.
            bboxs (list[dict]): bboxs info from the first stage

        Returns:
            list[list]: Decoded bbox, scores and labels after nms.
        """
        ret_list = []
        batch_size = batch_dict['batch_size']
        for batch_idx in range(batch_size):
            box_preds = batch_dict['batch_box_preds'][batch_idx]
            cls_preds = batch_dict['batch_cls_preds'][batch_idx]  # this is the predicted iou
            label_preds = batch_dict['roi_labels'][batch_idx]

            if box_preds.shape[-1] == 9:
                # move rotation to the end (the create submission file will take elements from 0:6 and -1)
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]]
            if self.roi_head_cfg['model_cfg']['LOSS_CONFIG']['CLS_LOSS'] == 'MSE':
                scores = torch.sqrt(cls_preds.reshape(-1)
                                    * batch_dict['roi_scores'][batch_idx].reshape(-1))
            else:
                scores = torch.sqrt(torch.sigmoid(cls_preds).reshape(-1)
                                   * batch_dict['roi_scores'][batch_idx].reshape(-1))
                # scores = torch.sqrt(batch_dict['roi_scores'][batch_idx].reshape(-1)
                #                    * batch_dict['roi_scores'][batch_idx].reshape(-1))
            mask = (label_preds != 0).view(-1)

            box_preds = box_preds[mask, :]
            scores = scores[mask]
            labels = label_preds[mask] - 1

            box_preds = img_metas[batch_idx]['box_type_3d'](
                box_preds, self.first_stage.bbox_coder.code_size)

            single_batch = [box_preds, scores, labels]

        ret_list.append(single_batch)

        return ret_list

    def forward_first_stage(self, pts_feats, img_metas, gt_bboxes_3d=None, gt_labels_3d=None):
        outs = self.first_stage(pts_feats)
        # merge multi_level features
        pts_feats = torch.cat([pts_feats], dim=1)
        bboxs = self.first_stage.get_bboxes(outs, img_metas)
        for i in range(len(bboxs)):
            tmp = {'bbox': bboxs[i][0], 'scores': bboxs[i][1],
                   'labels': bboxs[i][2]}
            bboxs[i] = tmp
        _ = multi_apply(self.first_stage.get_single_batch_sample_points,
                        bboxs, [self.num_points for _ in bboxs])

        _ = multi_apply(self.first_stage.get_single_bev_features, bboxs,
                        list(pts_feats))

        batched_rois = self.first_stage.prepare_batched_rois(bboxs)

        if self.training:
            first_stage_losses = self.first_stage.loss(gt_bboxes_3d,
                                                       gt_labels_3d, outs)
            batched_rois = self.add_gt_bbox(batched_rois,
                                            gt_bboxes_3d, gt_labels_3d)
            second_stage_losses = self.forward_second_stage(batched_rois,
                                                              bboxs, img_metas)

            return first_stage_losses.update(second_stage_losses)
        else:
            return self.forward_second_stage(batched_rois)

