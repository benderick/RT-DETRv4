"""D-FINE decoder extended with periodic oriented-box refinement."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..core import register
from .dfine_decoder import DFINETransformer, MLP, TransformerDecoder
from .dfine_utils import distance2bbox, weighting_function
from .rotated_denoising import get_rotated_contrastive_denoising_training_group
from .rotated_box_ops import regularize_rboxes
from .utils import inverse_sigmoid


class RotatedTransformerDecoder(TransformerDecoder):
    def forward(
        self, target, ref_points_unact, memory, spatial_shapes, bbox_head,
        angle_head, score_head, query_pos_head, pre_bbox_head, integral, up,
        reg_scale, attn_mask=None, memory_mask=None, dn_meta=None,
    ):
        output = target
        output_detach = pred_corners_undetach = angle_delta_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)
        boxes_out, logits_out, corners_out, refs_out = [], [], [], []
        project = weighting_function(self.reg_max, up, reg_scale) \
            if not hasattr(self, "project") else self.project
        ref_points_detach = torch.sigmoid(ref_points_unact)

        for index, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_position = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
            if index >= self.eval_idx + 1 and self.layer_scale > 1:
                query_position = F.interpolate(query_position, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_position.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_position.shape[-1])
                output_detach = output.detach()
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_position)
            if index == 0:
                pre_boxes = regularize_rboxes(
                    torch.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach)),
                    normalized_angle=True)
                pre_scores = score_head[0](output)
                initial_ref = pre_boxes.detach()

            pred_corners = bbox_head[index](output + output_detach) + pred_corners_undetach
            refined_xywh = distance2bbox(initial_ref[..., :4], integral(pred_corners, project), reg_scale)
            angle_delta = angle_head[index](output + output_detach) + angle_delta_undetach
            refined_angle = torch.remainder(initial_ref[..., 4:5] + 0.25 * torch.tanh(angle_delta), 1.0)
            refined_box = regularize_rboxes(
                torch.cat((refined_xywh, refined_angle), dim=-1), normalized_angle=True)

            if self.training or index == self.eval_idx:
                scores = self.lqe_layers[index](score_head[index](output), pred_corners)
                logits_out.append(scores)
                boxes_out.append(refined_box)
                corners_out.append(pred_corners)
                refs_out.append(initial_ref)
                if not self.training:
                    break
            pred_corners_undetach = pred_corners
            angle_delta_undetach = angle_delta
            ref_points_detach = refined_box.detach()
            output_detach = output.detach()
        return (torch.stack(boxes_out), torch.stack(logits_out), torch.stack(corners_out),
                torch.stack(refs_out), pre_boxes, pre_scores)


@register()
class RotatedDFINETransformer(DFINETransformer):
    """Preserve D-FINE FDR for ``cxcywh`` and iteratively refine ``theta``."""

    def __init__(
        self, num_classes=80, hidden_dim=256, num_queries=300,
        feat_channels=(512, 1024, 2048), feat_strides=(8, 16, 32),
        num_levels=3, num_points=4, nhead=8, num_layers=6,
        dim_feedforward=1024, dropout=0.0, activation="relu",
        num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0,
        learn_query_content=False, eval_spatial_size=None, eval_idx=-1,
        eps=1e-2, aux_loss=True, cross_attn_method="default",
        query_select_method="default", reg_max=32, reg_scale=4.0,
        layer_scale=1, mlp_act="relu",
    ):
        # Base initialization mutates feat_strides when extra levels are used;
        # keep configuration-owned lists immutable across repeated test builds.
        feat_channels, feat_strides = list(feat_channels), list(feat_strides)
        super().__init__(
            num_classes=num_classes, hidden_dim=hidden_dim, num_queries=num_queries,
            feat_channels=feat_channels, feat_strides=feat_strides, num_levels=num_levels,
            num_points=num_points, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            num_denoising=num_denoising, label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale, learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size, eval_idx=eval_idx, eps=eps,
            aux_loss=aux_loss, cross_attn_method=cross_attn_method,
            query_select_method=query_select_method, reg_max=reg_max,
            reg_scale=reg_scale, layer_scale=layer_scale, mlp_act=mlp_act)
        self.decoder.__class__ = RotatedTransformerDecoder
        hidden_dim = self.hidden_dim
        scaled_dim = round(self.decoder.layer_scale * hidden_dim)
        self.query_pos_head = MLP(5, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act)
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act)
        self.dec_angle_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 1, 3, act=mlp_act) for _ in range(self.eval_idx + 1)] +
            [MLP(scaled_dim, scaled_dim, 1, 3, act=mlp_act)
             for _ in range(self.num_layers - self.eval_idx - 1)])
        for head in (self.enc_bbox_head, self.pre_bbox_head, *self.dec_angle_head):
            init.constant_(head.layers[-1].weight, 0)
            init.constant_(head.layers[-1].bias, 0)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        if self.eval_spatial_size:
            anchors, valid = self._generate_anchors()
            self.anchors, self.valid_mask = anchors, valid

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05,
                          dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            eval_h, eval_w = self.eval_spatial_size
            spatial_shapes = [[int(eval_h / stride), int(eval_w / stride)]
                              for stride in self.feat_strides]
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
            xy = (torch.stack((grid_x, grid_y), -1).unsqueeze(0) + 0.5) / \
                torch.tensor([width, height], dtype=dtype, device=device)
            wh = torch.ones_like(xy) * grid_size * (2.0 ** level)
            angle = torch.full_like(xy[..., :1], 0.5)
            anchors.append(torch.cat((xy, wh, angle), -1).reshape(1, height * width, 5))
        anchors = torch.cat(anchors, dim=1)
        valid = ((anchors[..., :4] > self.eps) & (anchors[..., :4] < 1 - self.eps)).all(-1, keepdim=True)
        anchors = inverse_sigmoid(anchors)
        anchors = torch.where(valid, anchors, torch.inf)
        return anchors, valid

    def convert_to_deploy(self):
        super().convert_to_deploy()
        self.dec_angle_head = nn.ModuleList([
            self.dec_angle_head[i] if i <= self.eval_idx else nn.Identity()
            for i in range(len(self.dec_angle_head))])

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.training and self.num_denoising > 0:
            dn_logits, dn_boxes, attention_mask, dn_meta = \
                get_rotated_contrastive_denoising_training_group(
                    targets, self.num_classes, self.num_queries, self.denoising_class_embed,
                    self.num_denoising, self.label_noise_ratio, self.box_noise_scale)
        else:
            dn_logits = dn_boxes = attention_mask = dn_meta = None
        content, refs, enc_boxes, enc_logits = self._get_decoder_input(
            memory, spatial_shapes, dn_logits, dn_boxes)
        # Every model-facing box follows the same long-edge convention,
        # including encoder proposals and denoising references.
        refs = inverse_sigmoid(regularize_rboxes(
            torch.sigmoid(refs), normalized_angle=True).clamp(1e-5, 1 - 1e-5))
        enc_boxes = [regularize_rboxes(boxes, normalized_angle=True) for boxes in enc_boxes]
        out_boxes, out_logits, out_corners, out_refs, pre_boxes, pre_logits = self.decoder(
            content, refs, memory, spatial_shapes, self.dec_bbox_head, self.dec_angle_head,
            self.dec_score_head, self.query_pos_head, self.pre_bbox_head, self.integral,
            self.up, self.reg_scale, attn_mask=attention_mask, dn_meta=dn_meta)

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_boxes, pre_boxes = torch.split(pre_boxes, dn_meta["dn_num_split"], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_boxes, out_boxes = torch.split(out_boxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)

        if self.training:
            result = {"pred_logits": out_logits[-1], "pred_boxes": out_boxes[-1],
                      "pred_corners": out_corners[-1], "ref_points": out_refs[-1],
                      "up": self.up, "reg_scale": self.reg_scale}
        else:
            return {"pred_logits": out_logits[-1], "pred_boxes": out_boxes[-1]}
        if self.aux_loss:
            result["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1], out_boxes[:-1], out_corners[:-1], out_refs[:-1],
                out_corners[-1], out_logits[-1])
            result["enc_aux_outputs"] = self._set_aux_loss(enc_logits, enc_boxes)
            result["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_boxes}
            result["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}
            if dn_meta is not None:
                result["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits, dn_out_boxes, dn_out_corners, dn_out_refs,
                    dn_out_corners[-1], dn_out_logits[-1])
                result["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_boxes}
                result["dn_meta"] = dn_meta
        return result
