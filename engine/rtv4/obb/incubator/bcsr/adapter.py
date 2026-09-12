"""Boundary-conditioned spectral evidence for the existing OBB geometry heads.

No ground-truth boxes are used to sample or route features. The four sides form
an unordered set: shared operators and symmetric pooling preserve equivalent
rectangle parameterizations (half-turn and width/height exchange).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .....core import register


def boundary_points(boxes, height, width, samples_per_edge=4, offset_ratio=.15,
                    min_offset=1., max_offset=8.):
    """Normalized [B,Q,edge,sample,inside/outside,xy] grid and side descriptors.

Box dimensions are normalized independently by canvas width/height. Construct
geometry in pixels before rotating so non-square canvases remain correct.
Continuous image coordinates span [0,W] x [0,H]; grid_sample uses align_corners=False.
"""
    scale = boxes.new_tensor([width, height])
    center = boxes[..., :2] * scale
    size = boxes[..., 2:4] * scale
    angle = boxes[..., 4] * math.pi
    u = torch.stack((angle.cos(), angle.sin()), -1)
    v = torch.stack((-angle.sin(), angle.cos()), -1)
    normals = torch.stack((-v, u, v, -u), -2)
    tangents = torch.stack((u, v, -u, -v), -2)
    w, h = size.unbind(-1)
    lengths = torch.stack((w, h, w, h), -1)
    radii = torch.stack((h, w, h, w), -1) * .5
    midpoints = center.unsqueeze(-2) + normals * radii.unsqueeze(-1)
    positions = (torch.arange(samples_per_edge, device=boxes.device, dtype=boxes.dtype)+.5) / samples_per_edge - .5
    along = midpoints.unsqueeze(-2) + tangents.unsqueeze(-2) * lengths[..., None, None] * positions[:, None]
    offset = (size.amin(-1) * offset_ratio).clamp(min_offset, max_offset)
    signed = boxes.new_tensor([-1., 1.])
    points = along.unsqueeze(-2) + normals[..., None, None, :] * offset[..., None, None, None, None] * signed[:, None]
    # Global directions and normalized side length distinguish locations, with
    # no dependence on a side index or on a chosen long-edge chart.
    pose = torch.cat((normals, lengths.unsqueeze(-1) / math.sqrt(width*height)), -1)
    return points / scale, pose


@register()
class BoundarySpectralRefinement(nn.Module):
    def __init__(self, hidden_dim=256, bands=8, band_dim=8, routing_dim=32,
                 samples_per_edge=4, weighting="edge", sampling_reference="iterative",
                 offset_ratio=.15, min_offset=1., max_offset=8.):
        super().__init__()
        if weighting not in {"edge", "object"}:
            raise ValueError("weighting must be edge or object")
        if sampling_reference not in {"iterative", "initial"}:
            raise ValueError("sampling_reference must be iterative or initial")
        for name, value in dict(hidden_dim=hidden_dim, bands=bands, band_dim=band_dim,
                                routing_dim=routing_dim, samples_per_edge=samples_per_edge).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < min_offset <= max_offset or not 0 < offset_ratio <= .5:
            raise ValueError("Invalid boundary offset range")
        self.hidden_dim, self.bands = hidden_dim, bands
        self.band_dim, self.routing_dim = band_dim, routing_dim
        self.samples_per_edge = samples_per_edge
        self.weighting, self.sampling_reference = weighting, sampling_reference
        self.offset_ratio, self.min_offset, self.max_offset = offset_ratio, min_offset, max_offset
        # Identical weights process every band independently. Two even-kernel
        # reductions align feature centers with grid_sample's stride-4 lattice.
        # Registry injection builds the adapter before the decoder. Preserve
        # its RNG stream so paired baseline/adapter configs initialize every
        # shared parameter identically for the same seed.
        with torch.random.fork_rng(devices=[]):
            self.band_stem = nn.Sequential(
                nn.Conv2d(1, band_dim, 2, stride=2), nn.GELU(),
                nn.Conv2d(band_dim, band_dim, 2, stride=2), nn.GELU(),
                nn.Conv2d(band_dim, band_dim, 3, padding=1, groups=band_dim), nn.GELU())
            self.band_embedding = nn.Parameter(torch.empty(bands, routing_dim))
            nn.init.normal_(self.band_embedding, std=.02)
            descriptor_dim = 3 * band_dim
            self.key = nn.Linear(descriptor_dim, routing_dim)
            self.query = nn.Linear(hidden_dim, routing_dim)
            self.pose_key = nn.Linear(3, routing_dim)
            self.value = nn.Linear(descriptor_dim, hidden_dim)
            self.pose_value = nn.Linear(3, hidden_dim)
            self.readout = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            nn.init.zeros_(self.readout[-1].weight)
            nn.init.zeros_(self.readout[-1].bias)
        self.diagnostic_gradient_groups = {"boundary_spectral": "decoder.geometry_adapter."}

    def build_context(self, images, targets=None, diagnostics=False):
        if images.ndim != 4 or images.shape[1] != self.bands or not images.is_floating_point():
            raise ValueError(f"Expected floating [B,{self.bands},H,W] images")
        batch, bands, height, width = images.shape
        if height % 4 or width % 4:
            raise ValueError("BCSR canvas dimensions must be divisible by four")
        if targets is None:
            # Raw-image/profiler calls have no padded-region metadata.
            valid = torch.ones((batch,height,width), device=images.device, dtype=torch.bool)
        else:
            if len(targets) != batch or any("valid_mask" not in t for t in targets):
                raise ValueError("BCSR targets must provide one valid_mask per image")
            valid = torch.stack([t["valid_mask"].to(device=images.device, dtype=torch.bool) for t in targets])
            if valid.shape != (batch,height,width):
                raise ValueError("BCSR valid_mask must match the padded image canvas")
        # Values in masked regions cannot leak through the shallow convolutions.
        clean = images * valid.unsqueeze(1).to(images.dtype)
        features = self.band_stem(clean.reshape(batch*bands,1,height,width))
        return {"features": features, "valid": valid, "height": height, "width": width,
                "batch": batch, "diagnostics": bool(diagnostics), "records": []}

    def forward(self, queries, references, context, layer_index):
        if queries.shape[:2] != references.shape[:2] or references.shape[-1] != 5:
            raise ValueError("BCSR query/reference shape mismatch")
        if layer_index == 0:
            context["initial_references"] = references.detach()
        if "initial_references" not in context:
            raise ValueError("BCSR must start at decoder layer zero")
        boxes = context["initial_references"] if self.sampling_reference == "initial" else references.detach()
        points, pose = boundary_points(boxes, context["height"], context["width"],
            self.samples_per_edge, self.offset_ratio, self.min_offset, self.max_offset)
        batch, count = queries.shape[:2]
        features = context["features"]
        grid = points.reshape(batch, count*4*self.samples_per_edge*2, 1, 2) * 2 - 1
        band_grid = grid[:,None].expand(-1,self.bands,-1,-1,-1).reshape(batch*self.bands,-1,1,2)
        sampled = F.grid_sample(features, band_grid.to(features.dtype), mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        sampled = sampled.reshape(batch,self.bands,self.band_dim,count,4,self.samples_per_edge,2)
        sampled = sampled.permute(0,3,4,1,5,6,2)  # B,Q,edge,band,K,pair,D
        valid = F.grid_sample(context["valid"][:,None].to(features.dtype), grid.to(features.dtype),
                              mode="nearest", padding_mode="zeros", align_corners=False)
        valid = valid.reshape(batch,count,4,self.samples_per_edge,2) > .5
        valid = valid & ((points >= 0) & (points <= 1)).all(-1)
        # Require both inner and outer evidence at a sample position. Otherwise
        # padding could masquerade as a strong object boundary.
        paired = valid.all(-1)
        pair_weight = paired.to(sampled.dtype)
        denominator = pair_weight.sum(-1).clamp_min(1)
        pair_mean = (sampled * pair_weight[...,None,:,None,None]).sum(-3) / denominator[...,None,None,None]
        inside, outside = pair_mean.unbind(-2)
        descriptor = torch.cat((inside, outside, inside-outside), -1)
        edge_valid = paired.any(-1)
        keys = self.key(descriptor) + self.band_embedding
        query = self.query(queries).unsqueeze(-2) + self.pose_key(pose.to(queries.dtype))
        logits = (keys * query.unsqueeze(-2)).sum(-1) / math.sqrt(self.routing_dim)
        if self.weighting == "object":
            logits = (logits * edge_valid[...,None]).sum(-2,keepdim=True) / edge_valid.sum(-1,keepdim=True).clamp_min(1)[...,None]
            logits = logits.expand(-1,-1,4,-1)
        weights = logits.softmax(-1)
        evidence = (weights.unsqueeze(-1) * self.value(descriptor)).sum(-2)
        # Nonlinearity before set pooling preserves the association between
        # evidence and its physical side direction.
        evidence = F.gelu(evidence + self.pose_value(pose.to(evidence.dtype)))
        pooled = (evidence * edge_valid.unsqueeze(-1)).sum(-2) / edge_valid.sum(-1,keepdim=True).clamp_min(1)
        residual = self.readout(pooled) * edge_valid.any(-1,keepdim=True)
        if context["diagnostics"]:
            context["records"].append({
                "band_weights": weights.detach(), "sampling_points": points.detach(),
                "valid_fraction": paired.float().mean(-1).detach(),
                "residual_norm": residual.detach().float().norm(dim=-1),
                "input_reference": boxes.detach(),
            })
        return residual

    def diagnostics(self, context):
        records = context["records"]
        return {"schema_version": "bcsr-query-v1", **{
            name: torch.stack([r[name] for r in records])
            for name in records[0]
        }} if records else {"schema_version": "bcsr-query-v1"}
