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
from .obb.methods.o2.adr import (
    ADR_COMPONENT_NAMES,
    adr_orthogonality_error,
    adr_to_rbox,
    apply_adr_residuals,
    distribution_integral,
    o2_weighting_function,
)
from .rotated_denoising import get_rotated_contrastive_denoising_training_group
from .rotated_box_ops import regularize_rboxes
from .utils import inverse_sigmoid


class O2LocationQualityEstimator(nn.Module):
    """D-FINE's LQE generalized from four FDR sides to six ADR variables."""

    def __init__(self, components, k, hidden_dim, num_layers, reg_max, act="relu"):
        super().__init__()
        self.components = int(components)
        self.k = int(k)
        self.reg_max = int(reg_max)
        self.reg_conf = MLP(
            self.components * (self.k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, distributions):
        batch, length, _ = distributions.shape
        probability = F.softmax(
            distributions.reshape(batch, length, self.components, self.reg_max + 1), dim=-1)
        topk = probability.topk(self.k, dim=-1).values
        statistics = torch.cat((topk, topk.mean(dim=-1, keepdim=True)), dim=-1)
        return scores + self.reg_conf(statistics.reshape(batch, length, -1))


class RotatedTransformerDecoder(TransformerDecoder):
    def forward(
        self, target, ref_points_unact, memory, spatial_shapes, bbox_head,
        angle_head, score_head, query_pos_head, pre_bbox_head, integral, up,
        reg_scale, attn_mask=None, memory_mask=None, dn_meta=None,
        use_adr=False, adr_project=None,
        geometry_adapter=None, geometry_context=None,
    ):
        output = target
        output_detach = pred_corners_undetach = angle_delta_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)
        boxes_out, logits_out, corners_out, refs_out = [], [], [], []
        raw_logits_out, lqe_delta_out, input_refs_out = [], [], []
        project = weighting_function(self.reg_max, up, reg_scale) \
            if not hasattr(self, "project") else self.project
        ref_points_detach = torch.sigmoid(ref_points_unact)

        for index, layer in enumerate(self.layers):
            diagnostic_mode = bool(getattr(self, "diagnostic_mode", False))
            if diagnostic_mode:
                # This is the box that actually parameterizes the current
                # decoder layer's query position and rotated cross-attention.
                # It is deliberately distinct from ``initial_ref`` below,
                # which remains the fixed D-FINE refinement anchor.
                input_refs_out.append(ref_points_detach)
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

            # Both supported variants retain D-FINE's fixed-anchor,
            # cumulative-logit refinement semantics.
            geometry_output = output
            if geometry_adapter is not None:
                residual = geometry_adapter(output, ref_points_detach, geometry_context, index)
                if residual.shape != output.shape:
                    raise ValueError("Geometry adapter residual must have the same shape as query features")
                geometry_output = output + residual
            pred_corners = (
                bbox_head[index](geometry_output + output_detach)
                + pred_corners_undetach
            )
            if use_adr:
                if adr_project is None:
                    raise RuntimeError("ADR decoder requires its non-uniform distribution project")
                residuals = distribution_integral(pred_corners, adr_project, components=6)
                refined_box = regularize_rboxes(
                    adr_to_rbox(initial_ref, residuals, normalized_angle=True),
                    normalized_angle=True)
            else:
                refined_xywh = distance2bbox(
                    initial_ref[..., :4], integral(pred_corners, project), reg_scale)
                angle_delta = angle_head[index](geometry_output + output_detach) + angle_delta_undetach
                refined_angle = torch.remainder(
                    initial_ref[..., 4:5] + 0.25 * torch.tanh(angle_delta), 1.0)
                refined_box = regularize_rboxes(
                    torch.cat((refined_xywh, refined_angle), dim=-1), normalized_angle=True)

            if self.training or diagnostic_mode or index == self.eval_idx:
                raw_scores = score_head[index](output)
                scores = self.lqe_layers[index](raw_scores, pred_corners)
                logits_out.append(scores)
                if diagnostic_mode:
                    raw_logits_out.append(raw_scores)
                    lqe_delta_out.append(scores - raw_scores)
                boxes_out.append(refined_box)
                corners_out.append(pred_corners)
                refs_out.append(initial_ref)
                # Diagnostic evaluation keeps the earlier layer outputs but
                # must stop at the configured evaluation layer, exactly like
                # ordinary inference (eval_idx need not be the last layer).
                if not self.training and index == self.eval_idx:
                    break
            pred_corners_undetach = pred_corners
            if not use_adr:
                angle_delta_undetach = angle_delta
            ref_points_detach = refined_box.detach()
            output_detach = output.detach()
        return (
            torch.stack(boxes_out),
            torch.stack(logits_out),
            torch.stack(corners_out),
            torch.stack(refs_out),
            pre_boxes,
            pre_scores,
            torch.stack(raw_logits_out) if raw_logits_out else None,
            torch.stack(lqe_delta_out) if lqe_delta_out else None,
            torch.stack(input_refs_out) if input_refs_out else None,
        )


@register()
class RotatedDFINETransformer(DFINETransformer):
    """D-FINE OBB decoder supporting direct-angle and O² ADR refinement."""

    __inject__ = ["denoising_builder", "geometry_adapter"]

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
        adr_a=0.5, adr_c=0.25,
        refinement_mode="o2_adr",
        ocd_mode="box", ocd_lambda1=1.0, ocd_lambda2=2.0,
        ocd_lambda3=9.0, ocd_lambda4=18.0,
        ocd_lambda5=0.3, ocd_lambda6=0.6,
        ocd_crowded_policy="strict_budget_random",
        denoising_builder=None,
        geometry_adapter=None,
        box_coordinate_mode="per_axis",
    ):
        # Base initialization mutates feat_strides when extra levels are used;
        # keep configuration-owned lists immutable across repeated test builds.
        if box_coordinate_mode not in {"per_axis", "isotropic"}:
            raise ValueError("Unknown box_coordinate_mode")
        self.box_coordinate_mode = box_coordinate_mode
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
        self.decoder.diagnostic_mode = False
        self.decoder.diagnostic_attention_mode = False
        for layer in self.decoder.layers:
            layer.cross_attn.box_coordinate_mode = box_coordinate_mode
        refinement_mode = str(refinement_mode)
        supported_modes = {"direct_angle", "o2_adr"}
        if refinement_mode not in supported_modes:
            raise ValueError(
                f"Unknown refinement_mode {refinement_mode!r}; expected one of "
                f"{sorted(supported_modes)}")
        self.refinement_mode = refinement_mode
        self.use_adr = refinement_mode == "o2_adr"
        self.ocd_mode = str(ocd_mode)
        self.ocd_crowded_policy = str(ocd_crowded_policy)
        if denoising_builder is not None and not callable(denoising_builder):
            raise TypeError("denoising_builder must be callable or None")
        self.denoising_builder = denoising_builder
        if geometry_adapter is not None and not isinstance(geometry_adapter, nn.Module):
            raise TypeError("geometry_adapter must be a torch module or None")
        if geometry_adapter is not None and layer_scale != 1:
            raise ValueError("Geometry adapters currently require layer_scale=1")
        self.geometry_adapter = geometry_adapter
        if geometry_adapter is not None and getattr(geometry_adapter, "box_coordinate_mode", box_coordinate_mode) != box_coordinate_mode:
            raise ValueError("Geometry adapter and decoder box coordinates must agree")
        self.ocd_lambdas = (
            float(ocd_lambda1), float(ocd_lambda2), float(ocd_lambda3),
            float(ocd_lambda4), float(ocd_lambda5), float(ocd_lambda6),
        )
        hidden_dim = self.hidden_dim
        scaled_dim = round(self.decoder.layer_scale * hidden_dim)
        self.query_pos_head = MLP(5, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act)
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act)
        if self.use_adr:
            self.dec_bbox_head = nn.ModuleList(
                [MLP(hidden_dim, hidden_dim, 6 * (self.reg_max + 1), 3, act=mlp_act)
                 for _ in range(self.eval_idx + 1)] +
                [MLP(scaled_dim, scaled_dim, 6 * (self.reg_max + 1), 3, act=mlp_act)
                 for _ in range(self.num_layers - self.eval_idx - 1)])
            self.dec_angle_head = nn.ModuleList([nn.Identity() for _ in range(self.num_layers)])
            self.decoder.lqe_layers = nn.ModuleList([
                O2LocationQualityEstimator(6, 4, 64, 2, self.reg_max, act=activation)
                for _ in range(self.num_layers)
            ])
            self.register_buffer(
                "adr_project", o2_weighting_function(self.reg_max, adr_a, adr_c))
        else:
            # Direct-angle is the explicit scalar-angle architectural control.
            self.dec_angle_head = nn.ModuleList(
                [MLP(hidden_dim, hidden_dim, 1, 3, act=mlp_act)
                 for _ in range(self.eval_idx + 1)] +
                [MLP(scaled_dim, scaled_dim, 1, 3, act=mlp_act)
                 for _ in range(self.num_layers - self.eval_idx - 1)])
        initialized_heads = [self.enc_bbox_head, self.pre_bbox_head]
        initialized_heads.extend(
            head for head in self.dec_angle_head if hasattr(head, "layers"))
        initialized_heads.extend(self.dec_bbox_head if self.use_adr else [])
        for head in initialized_heads:
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
            if self.box_coordinate_mode == "isotropic":
                # Grid centers in an isotropic model plane; proposal side
                # lengths below are fractions of the longest canvas side.
                xy = xy * xy.new_tensor([width, height]) / max(width, height)
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
        if not self.use_adr:
            self.dec_angle_head = nn.ModuleList([
                self.dec_angle_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_angle_head))])

    def set_diagnostic_mode(self, enabled=True, capture_attention=None):
        """Expose per-decoder-layer predictions during evaluation.

        This is deliberately opt-in so deployment and ordinary inference keep
        exactly the same output contract and memory footprint.  Layer outputs
        and cross-attention traces are controlled separately because the latter
        are much larger and are needed only for a deterministic image subset.
        """
        enabled = bool(enabled)
        if capture_attention is None:
            capture_attention = enabled
        capture_attention = bool(capture_attention) and enabled
        self.decoder.diagnostic_mode = enabled
        self.decoder.diagnostic_attention_mode = capture_attention
        for layer in self.decoder.layers:
            layer.cross_attn.diagnostic_mode = capture_attention
        return self

    def _decode_queries(
        self,
        content,
        refs,
        memory,
        spatial_shapes,
        attention_mask,
        dn_meta,
        context=None,
    ):
        """Run the shared direct-angle/O² decoder."""

        return self.decoder(
            content, refs, memory, spatial_shapes, self.dec_bbox_head,
            self.dec_angle_head, self.dec_score_head, self.query_pos_head,
            self.pre_bbox_head, self.integral, self.up, self.reg_scale,
            attn_mask=attention_mask, dn_meta=dn_meta,
            use_adr=self.use_adr,
            adr_project=self.adr_project if self.use_adr else None,
            geometry_adapter=self.geometry_adapter,
            geometry_context=context,
        )

    def _build_denoising_group(self, targets):
        """Build training-only DN queries through one explicit strategy seam.

        The default path is the established O²/direct-angle implementation.
        Incubator methods may inject a callable without adding a refinement
        mode, changing detector parameters, or duplicating decoder forward.
        """

        arguments = dict(
            targets=targets,
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            class_embed=self.denoising_class_embed,
            num_denoising=self.num_denoising,
            label_noise_ratio=self.label_noise_ratio,
            box_noise_scale=self.box_noise_scale,
            mode=self.ocd_mode,
            lambda1=self.ocd_lambdas[0],
            lambda2=self.ocd_lambdas[1],
            lambda3=self.ocd_lambdas[2],
            lambda4=self.ocd_lambdas[3],
            lambda5=self.ocd_lambdas[4],
            lambda6=self.ocd_lambdas[5],
            crowded_policy=self.ocd_crowded_policy,
        )
        builder = self.denoising_builder
        return (
            get_rotated_contrastive_denoising_training_group(**arguments)
            if builder is None else builder(**arguments)
        )

    def build_context(self, images, targets=None):
        if self.geometry_adapter is None:
            return None
        return self.geometry_adapter.build_context(images, targets,
            diagnostics=bool(self.decoder.diagnostic_mode))

    def forward(self, feats, targets=None, context=None):
        if self.geometry_adapter is not None and context is None:
            raise ValueError("An enabled geometry adapter requires image context from RTv4")
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.training and self.num_denoising > 0:
            dn_logits, dn_boxes, attention_mask, dn_meta = \
                self._build_denoising_group(targets)
        else:
            dn_logits = dn_boxes = attention_mask = dn_meta = None
        content, refs, enc_boxes, enc_logits = self._get_decoder_input(
            memory, spatial_shapes, dn_logits, dn_boxes)
        decoded_refs = torch.sigmoid(refs)
        if self.use_adr and dn_meta is not None:
            # Released O² box noise keeps theta fixed while perturbing the
            # two xyxy vertices.  If the perturbed w/h cross, canonicalizing
            # this DN tuple here would numerically add a quarter turn before
            # the first attention layer.  Preserve the released DN reference;
            # matching-query references retain the framework's long-edge
            # convention.  The first traditional OBB head canonicalizes both.
            dn_count = dn_meta["dn_num_split"][0]
            dn_refs, matching_refs = torch.split(
                decoded_refs, [dn_count, decoded_refs.shape[1] - dn_count], dim=1)
            decoded_refs = torch.cat((
                dn_refs,
                regularize_rboxes(matching_refs, normalized_angle=True),
            ), dim=1)
        else:
            # This branch is unchanged for the direct-angle control.
            decoded_refs = regularize_rboxes(
                decoded_refs, normalized_angle=True)
        refs = inverse_sigmoid(decoded_refs.clamp(1e-5, 1 - 1e-5))
        enc_boxes = [regularize_rboxes(boxes, normalized_angle=True) for boxes in enc_boxes]
        (out_boxes, out_logits, out_corners, out_refs, pre_boxes, pre_logits,
         out_raw_logits, out_lqe_delta, out_input_refs) = \
            self._decode_queries(
            content, refs, memory, spatial_shapes, attention_mask, dn_meta, context=context
        )

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
            result["distribution_project"] = (
                self.adr_project if self.use_adr else
                weighting_function(self.reg_max, self.up, self.reg_scale))
            result["distribution_names"] = (
                ADR_COMPONENT_NAMES if self.use_adr else
                ("left", "top", "right", "bottom"))
            result["refinement_kind"] = (
                "adr" if self.use_adr else "fdr_angle")
            result["refinement_mode"] = self.refinement_mode
            if self.use_adr:
                result["adr_project"] = self.adr_project
                result["adr_geometry_contract"] = "dfine4_plus_vertex2_equal_diagonal"
        else:
            result = {"pred_logits": out_logits[-1], "pred_boxes": out_boxes[-1]}
            if bool(getattr(self.decoder, "diagnostic_mode", False)):
                result["diagnostic_pre_logits"] = pre_logits
                result["diagnostic_pre_boxes"] = pre_boxes
                result["diagnostic_layer_logits"] = out_logits
                result["diagnostic_layer_class_logits_before_lqe"] = out_raw_logits
                result["diagnostic_layer_lqe_logit_delta"] = out_lqe_delta
                result["diagnostic_layer_boxes"] = out_boxes
                result["diagnostic_layer_anchors"] = out_refs
                result["diagnostic_layer_input_refs"] = out_input_refs
                result["diagnostic_layer_distributions"] = out_corners
                result["diagnostic_refinement_kind"] = (
                    "adr" if self.use_adr else "fdr_angle")
                result["diagnostic_refinement_mode"] = self.refinement_mode
                result["diagnostic_distribution_project"] = (
                    self.adr_project if self.use_adr else
                    weighting_function(self.reg_max, self.up, self.reg_scale))
                result["diagnostic_distribution_names"] = (
                    ADR_COMPONENT_NAMES if self.use_adr else
                    ("left", "top", "right", "bottom"))
                if self.use_adr:
                    adr_residuals = distribution_integral(
                        out_corners, self.adr_project, components=6)
                    adr_values = apply_adr_residuals(out_refs, adr_residuals)
                    result["diagnostic_layer_adr_residuals"] = adr_residuals
                    result["diagnostic_layer_adr_values"] = adr_values
                    result["diagnostic_layer_adr_raw_orthogonality_error"] = \
                        adr_orthogonality_error(adr_values)
                    result["diagnostic_adr_geometry_contract"] = \
                        "dfine4_plus_vertex2_equal_diagonal"
                active_layers = self.decoder.layers[:len(out_boxes)]
                if bool(getattr(self.decoder, "diagnostic_attention_mode", False)) and \
                        active_layers and all(
                        hasattr(layer.cross_attn, "last_sampling_locations")
                        for layer in active_layers):
                    result["diagnostic_sampling_locations"] = torch.stack([
                        layer.cross_attn.last_sampling_locations for layer in active_layers])
                    result["diagnostic_sampling_unrotated_offsets"] = torch.stack([
                        layer.cross_attn.last_unrotated_offsets for layer in active_layers])
                    result["diagnostic_sampling_rotated_offsets"] = torch.stack([
                        layer.cross_attn.last_rotated_offsets for layer in active_layers])
                    result["diagnostic_sampling_attention_weights"] = torch.stack([
                        layer.cross_attn.last_attention_weights for layer in active_layers])
                    result["diagnostic_sampling_points_per_level"] = tuple(
                        active_layers[0].cross_attn.num_points_list)
            if self.geometry_adapter is not None and self.decoder.diagnostic_mode:
                result["diagnostic_query_extensions"] = self.geometry_adapter.diagnostics(context)
            return result
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
                if "method_diagnostics" in dn_meta:
                    result["method_train_diagnostics"] = \
                        dn_meta["method_diagnostics"]
        return result
