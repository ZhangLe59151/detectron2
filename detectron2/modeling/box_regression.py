# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Tuple
import torch
import pdb, json
import numpy as np
import torchvision.ops as ops
# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


__all__ = ["Box2BoxTransform", "Box2BoxTransformRotated"]

@torch.jit.script
class Box2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is parameterized
    by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(
        self, weights: Tuple[float, float, float, float], 
        scale_clamp: float = _DEFAULT_SCALE_CLAMP
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp
        self.area = weights[2] * weights[3]

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        return deltas
    
    def get_relative_areas(self, src_boxes_list, target_boxes_list, areas):
        # assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        # assert isinstance(target_boxes, torch.Tensor), type(target_boxes)
        #get source weight height
        assert isinstance(src_boxes_list, tuple)
        assert isinstance(target_boxes_list, tuple)

        sas = []
        tas = []
        for i in range(len(src_boxes_list)):
            src_boxes = src_boxes_list[i]   
            target_boxes = target_boxes_list[i]

            src_widths = src_boxes[:, 2] - src_boxes[:, 0]
            src_heights = src_boxes[:, 3] - src_boxes[:, 1]
            #get target weight height
            target_widths = target_boxes[:, 2] - target_boxes[:, 0]
            target_heights = target_boxes[:, 3] - target_boxes[:, 1]
            #caluate source area
            sas.append(src_widths * src_heights / areas[i])
            tas.append(target_widths * target_heights / areas[i])

        source_area = torch.cat(sas, dim=0)
        target_area =  torch.cat(tas, dim=0)
        '''
        target_x = target_boxes[:, 0]
        target_y = target_boxes[:, 1]
        tar_x_1 ,tar_x_2, tar_x_3, tar_x_4 = target_x.split([64,64,64,64], dim=0)
        tar_y_1 ,tar_y_2, tar_y_3, tar_y_4 = target_y.split([64,64,64,64], dim=0)
        p_box_1 = tar_x_1 + tar_y_1 * 0.001
        p_box_2 = tar_x_2 + tar_y_2 * 0.001
        p_box_3 = tar_x_3 + tar_y_3 * 0.001
        p_box_4 = tar_x_4 + tar_y_4 * 0.001
        # print('pbox_1 :', p_box_1)
        zero = torch.zeros_like(p_box_1)
        p_box_new = p_box_1
        '''
        # assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        # return deltas
        return source_area, target_area
    
    def iou_of_different_predictions(self, src_boxes_list, fg_inds_list, gt_sampled_targets_list):
        assert isinstance(src_boxes_list, tuple)
        prediction1 = []
        prediction2 = []
        for i in range(len(src_boxes_list)):
            src_boxes = src_boxes_list[i]
            fg_inds = fg_inds_list[i]
            gt_sampled_targets = gt_sampled_targets_list[i]
            src_boxes = src_boxes[fg_inds]
            gt_sampled_targets = gt_sampled_targets[fg_inds]
            src_boxes = torch.cat([src_boxes, gt_sampled_targets], dim=1)
            for m in range(len(gt_sampled_targets)):
                for n in range(len(gt_sampled_targets)):
                    prediction1.append(src_boxes[m])
                    prediction2.append(src_boxes[n])

        prediction1 = torch.cat(prediction1, dim=0)
        prediction2 = torch.cat(prediction2, dim=0)
        ignored_pairs = prediction1.eq(prediction2)
        prediction1 = prediction1[ignored_pairs]
        prediction2 = prediction2[ignored_pairs]

        return ops.boxes.box_iou(prediction1[:, :4], prediction2[:, :4])

    def get_relative_areas_ratio_1(self, src_boxes_list, target_boxes_list, areas, pred_class_logits, gt_classes, gt_sampled_targets):
        # assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        # assert isinstance(target_boxes, torch.Tensor), type(target_boxes)
        #get source weight height
        assert isinstance(src_boxes_list, tuple)
        assert isinstance(target_boxes_list, tuple)

        # print('pred_class_logits', pred_class_logits)
        # print('gt_classes', gt_classes)
        # print('gt_sampled_targets', gt_sampled_targets)

        number_of_target_box = 0
        target_box = []

        # tar_x_1 ,tar_x_2, tar_x_3, tar_x_4 = target_boxes_list.split([64,64,64,64], dim=0)
        for item in target_boxes_list[0]:
            need_add = True
            for item_box in target_box:
                if item_box.equal(item):
                    need_add = False
            if need_add:
                target_box.append(item)
                number_of_target_box += 1
        area_0 = 0
        for item in target_box:
            area_0 = area_0 + (item[2] - item[0]) * (item[3] - item[1])
        ratio_0 = area_0 / areas[0]
        print('ratio area', ratio_0)
        
        score_num = 0
        area_s = 0
        for box in target_box:
            i = 0
            area_t = 0
            for item in target_boxes_list[0]:
                if box.equal(item):
                    if (torch.gt(pred_class_logits[i][0], score_num) and torch.gt(pred_class_logits[i][0], pred_class_logits[i][1])):
                        score_num = pred_class_logits[i][0]
                        area_t = (src_boxes_list[0][i][2] - src_boxes_list[0][i][0]) * (src_boxes_list[0][i][3] - src_boxes_list[0][i][1])
                i += 1
            area_s = area_s + area_t
        area_0_pre = area_s/ areas[0]
        print(area_0_pre)

        sas = []
        tas = []
        for i in range(len(src_boxes_list)):
            src_boxes = src_boxes_list[i]   
            target_boxes = target_boxes_list[i]

            src_widths = src_boxes[:, 2] - src_boxes[:, 0]
            src_heights = src_boxes[:, 3] - src_boxes[:, 1]
            #get target weight height
            target_widths = target_boxes[:, 2] - target_boxes[:, 0]
            target_heights = target_boxes[:, 3] - target_boxes[:, 1]
            #caluate source area
            sas.append(src_widths * src_heights / areas[i])
            tas.append(target_widths * target_heights / areas[i])

        source_area = torch.cat(sas, dim=0)
        target_area =  torch.cat(tas, dim=0)

        # assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        # return deltas
        return source_area, target_area

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2
        return pred_boxes

@torch.jit.script
class Box2BoxTransformRotated(object):
    """
    The box-to-box transform defined in Rotated R-CNN. The transformation is parameterized
    by 5 deltas: (dx, dy, dw, dh, da). The transformation scales the box's width and height
    by exp(dw), exp(dh), shifts a box's center by the offset (dx * width, dy * height),
    and rotate a box's angle by da (radians).
    Note: angles of deltas are in radians while angles of boxes are in degrees.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float, float],
        scale_clamp: float = _DEFAULT_SCALE_CLAMP,
    ):
        """
        Args:
            weights (5-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh, da) deltas. These are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh, da) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): Nx5 source boxes, e.g., object proposals
            target_boxes (Tensor): Nx5 target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_ctr_x, src_ctr_y, src_widths, src_heights, src_angles = torch.unbind(src_boxes, dim=1)

        target_ctr_x, target_ctr_y, target_widths, target_heights, target_angles = torch.unbind(
            target_boxes, dim=1
        )

        wx, wy, ww, wh, wa = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)
        # Angles of deltas are in radians while angles of boxes are in degrees.
        # the conversion to radians serve as a way to normalize the values
        da = target_angles - src_angles
        da = (da + 180.0) % 360.0 - 180.0  # make it in [-180, 180)
        da *= wa * math.pi / 180.0

        deltas = torch.stack((dx, dy, dw, dh, da), dim=1)
        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransformRotated are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh, da) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*5).
                deltas[i] represents box transformation for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 5)
        """
        assert deltas.shape[1] % 5 == 0 and boxes.shape[1] == 5

        boxes = boxes.to(deltas.dtype).unsqueeze(2)

        ctr_x = boxes[:, 0]
        ctr_y = boxes[:, 1]
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        angles = boxes[:, 4]

        wx, wy, ww, wh, wa = self.weights

        dx = deltas[:, 0::5] / wx
        dy = deltas[:, 1::5] / wy
        dw = deltas[:, 2::5] / ww
        dh = deltas[:, 3::5] / wh
        da = deltas[:, 4::5] / wa

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::5] = dx * widths + ctr_x  # x_ctr
        pred_boxes[:, 1::5] = dy * heights + ctr_y  # y_ctr
        pred_boxes[:, 2::5] = torch.exp(dw) * widths  # width
        pred_boxes[:, 3::5] = torch.exp(dh) * heights  # height

        # Following original RRPN implementation,
        # angles of deltas are in radians while angles of boxes are in degrees.
        pred_angle = da * 180.0 / math.pi + angles
        pred_angle = (pred_angle + 180.0) % 360.0 - 180.0  # make it in [-180, 180)

        pred_boxes[:, 4::5] = pred_angle

        return pred_boxes
