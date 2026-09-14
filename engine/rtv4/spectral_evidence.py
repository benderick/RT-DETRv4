"""Instance-conditioned spectral evidence, without inference-time annotations.

One query update combines object compatibility and background-relative contrast.
The covariance direction is a classical statistical discriminant, not a new
physical spectral model. Scores and gates are not calibrated probabilities.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..core import register


def sampling_support(boxes, height, width, candidate_side=7, background_side=11,
                     context_scale=3., min_context_px=32., box_coordinate_mode="isotropic"):
    """Return candidate/background grids in [0,1] canvas coordinates and BG mask.

    Boxes use a half-turn angle and declared length normalization. The wider
    square background includes corners outside the object's narrow OBB ring.
    All geometry is constructed in pixels before grid_sample conversion.
    """
    canvas = boxes.new_tensor([width, height])
    scale = boxes.new_full((2,), max(width, height)) if box_coordinate_mode == "isotropic" else canvas
    center, size = boxes[..., :2] * scale, boxes[..., 2:4].clamp_min(1e-5) * scale
    angle = boxes[..., 4] * math.pi
    u = torch.stack((angle.cos(), angle.sin()), -1)
    v = torch.stack((-angle.sin(), angle.cos()), -1)
    position = (torch.arange(candidate_side, device=boxes.device, dtype=boxes.dtype) + .5) / candidate_side - .5
    yy, xx = torch.meshgrid(position, position, indexing="ij")
    xy = torch.stack((xx.flatten(), yy.flatten()), -1)
    local = 1.2 * size.unsqueeze(-2) * xy
    candidates = center.unsqueeze(-2) + local[..., :1]*u.unsqueeze(-2) + local[..., 1:]*v.unsqueeze(-2)
    position = (torch.arange(background_side, device=boxes.device, dtype=boxes.dtype) + .5) / background_side - .5
    yy, xx = torch.meshgrid(position, position, indexing="ij")
    xy = torch.stack((xx.flatten(), yy.flatten()), -1)
    span = (size.amax(-1) * context_scale).clamp_min(min_context_px)
    offset = span[..., None, None] * xy
    background = center.unsqueeze(-2) + offset
    along_u = (offset*u.unsqueeze(-2)).sum(-1).abs()
    along_v = (offset*v.unsqueeze(-2)).sum(-1).abs()
    outside = (along_u > .65*size[..., 0, None]) | (along_v > .65*size[..., 1, None])
    return candidates/canvas, background/canvas, outside


def background_statistics(values, weights, shrinkage=.2, ridge_ratio=.01, ridge_floor=1e-3):
    """Weighted mean, regularized covariance and effective support count (FP32)."""
    total = weights.sum(-1).clamp_min(1e-8)
    mean = (values*weights.unsqueeze(-1)).sum(-2)/total.unsqueeze(-1)
    centered = values-mean.unsqueeze(-2)
    squares = weights.square().sum(-1)
    denominator = (total-squares/total).clamp_min(1e-8)
    covariance = (centered.transpose(-1,-2) @ (centered*weights.unsqueeze(-1)))/denominator[...,None,None]
    diagonal = covariance.diagonal(dim1=-2,dim2=-1)
    ridge = ridge_ratio*diagonal.mean(-1).clamp_min(0)+ridge_floor
    regular = (1-shrinkage)*covariance + torch.diag_embed(shrinkage*diagonal+ridge.unsqueeze(-1))
    effective = total.square()/squares.clamp_min(1e-8)
    return mean, regular, effective


def masked_softmax(logits, valid):
    weights = logits.masked_fill(~valid, -1e4).softmax(-1)*valid
    return weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)


@register()
class BackgroundConditionedSpectralEvidence(nn.Module):
    """Opt-in query refinement at exactly one decoder layer.

    Modes share parameter shapes, initialization, samples and computation:
    plain / diagonal / no_compatibility / full. Disabled factors remain in the
    autograd graph with zero coefficients, so ordinary DDP needs no unused-
    parameter discovery. Their zero gradients are intentional controls.
    """
    def __init__(self, hidden_dim=256, bands=8, band_dim=4, spectral_dim=8,
                 mode="full", apply_layer=1, candidate_side=7, background_side=11,
                 context_scale=3., min_context_px=32., min_background_support=16.,
                 shrinkage=.2, ridge_ratio=.01, ridge_floor=1e-3,
                 temperature=1., residual_scale=.1, query_chunk_size=64,
                 box_coordinate_mode="isotropic"):
        super().__init__()
        if mode not in {"plain", "diagonal", "no_compatibility", "full"}:
            raise ValueError("Unknown spectral evidence mode")
        for name, value in dict(hidden_dim=hidden_dim,bands=bands,band_dim=band_dim,
                spectral_dim=spectral_dim,candidate_side=candidate_side,
                background_side=background_side,query_chunk_size=query_chunk_size).items():
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(apply_layer,bool) or not isinstance(apply_layer,int) or apply_layer<0:
            raise ValueError("apply_layer must be a nonnegative integer")
        if not math.isfinite(shrinkage) or not 0<=shrinkage<=1:
            raise ValueError("Invalid regularization")
        if not all(math.isfinite(x) and x>0 for x in (ridge_floor,temperature,min_context_px,residual_scale,min_background_support)):
            raise ValueError("Scales and support must be finite and positive")
        if not math.isfinite(ridge_ratio) or ridge_ratio<0 or not math.isfinite(context_scale) or context_scale<=1.3:
            raise ValueError("Invalid covariance ridge or background extent")
        if box_coordinate_mode not in {"per_axis","isotropic"}:
            raise ValueError("Unknown box_coordinate_mode")
        self.hidden_dim,self.bands,self.spectral_dim = hidden_dim,bands,spectral_dim
        self.mode,self.apply_layer = mode,apply_layer
        self.candidate_side,self.background_side = candidate_side,background_side
        self.context_scale,self.min_context_px = context_scale,min_context_px
        self.min_background_support = min_background_support
        self.shrinkage,self.ridge_ratio,self.ridge_floor = shrinkage,ridge_ratio,ridge_floor
        self.temperature,self.residual_scale = temperature,residual_scale
        self.query_chunk_size,self.box_coordinate_mode = query_chunk_size,box_coordinate_mode
        # Diagnostic-only intervention, never in training configs/checkpoints.
        self.background_override = None
        # Registry injection creates this module before the decoder. Preserve
        # the RNG sequence used to initialize all shared baseline parameters.
        with torch.random.fork_rng(devices=[]):
            # Reduce before expanding channels: memory stays practical for an
            # eight-band, full-canvas batch. Groups preserve band provenance.
            self.band_stem = nn.Sequential(
                nn.Conv2d(bands,bands*band_dim,3,padding=1,groups=bands),nn.GELU(),
                nn.Conv2d(bands*band_dim,bands*band_dim,3,padding=1,groups=bands),nn.GELU())
            self.embedding = nn.Conv2d(bands*band_dim,spectral_dim,1)
            self.query_norm = nn.LayerNorm(hidden_dim)
            self.target = nn.Linear(hidden_dim,spectral_dim)
            self.compatibility_scale = nn.Linear(hidden_dim,spectral_dim)
            self.semantic_query = nn.Linear(hidden_dim,spectral_dim)
            self.value = nn.Linear(spectral_dim,hidden_dim)
            self.readout = nn.Linear(hidden_dim,hidden_dim,bias=False)
            nn.init.zeros_(self.compatibility_scale.weight)
            nn.init.zeros_(self.compatibility_scale.bias)
            nn.init.zeros_(self.readout.weight)
        self.diagnostic_gradient_groups = {"spectral_evidence":"decoder.query_adapter."}

    def build_context(self, images, targets=None, diagnostics=False):
        if images.ndim!=4 or images.shape[1]!=self.bands or not images.is_floating_point():
            raise ValueError(f"Expected floating [B,{self.bands},H,W] images")
        batch,_,height,width = images.shape
        if height%4 or width%4:
            raise ValueError("Spectral branch requires canvas dimensions divisible by four")
        if targets is None:
            valid = torch.ones((batch,height,width),device=images.device,dtype=torch.bool)
        else:
            if len(targets)!=batch or any("valid_mask" not in t for t in targets):
                raise ValueError("Provide image valid_mask metadata, without requiring boxes or classes")
            valid = torch.stack([t["valid_mask"].to(device=images.device,dtype=torch.bool) for t in targets])
            if valid.shape!=(batch,height,width):
                raise ValueError("valid_mask must match the image canvas")
        clean = images*valid.unsqueeze(1)
        pooled = F.avg_pool2d(clean,4)
        features = self.embedding(self.band_stem(pooled))
        return {"features":features,"valid":valid,"height":height,"width":width,
                "diagnostics":bool(diagnostics),"records":{},"visited_layers":0}

    @staticmethod
    def sample(feature, points, mode="bilinear"):
        batch,count,samples,_ = points.shape
        grid = points.reshape(batch,count*samples,1,2)*2-1
        values = F.grid_sample(feature.float(),grid.float(),mode=mode,padding_mode="zeros",align_corners=False)
        return values.squeeze(-1).transpose(1,2).reshape(batch,count,samples,-1)

    def _chunk(self, queries, references, context):
        points,bg_points,outside = sampling_support(references.detach().float(),context["height"],context["width"],
            self.candidate_side,self.background_side,self.context_scale,self.min_context_px,self.box_coordinate_mode)
        # Compare relative spectral patterns in the same bounded space. Without
        # this, unconstrained prototype norms can close the exponential gate
        # during the paper recipe's high bias-LR warmup, before learning it.
        z = F.normalize(self.sample(context["features"],points),dim=-1,eps=1e-6)
        bg = F.normalize(self.sample(context["features"],bg_points),dim=-1,eps=1e-6)
        valid = (self.sample(context["valid"][:,None],points,"nearest").squeeze(-1)>.5)&((points>=0)&(points<=1)).all(-1)
        bg_valid = (self.sample(context["valid"][:,None],bg_points,"nearest").squeeze(-1)>.5)&((bg_points>=0)&(bg_points<=1)).all(-1)&outside
        # P3 encoder objectness precedes DN creation; matching queries cannot
        # see GT-dependent DN boxes through this exclusion mask.
        objectness = self.sample(context["encoder_objectness"],bg_points).squeeze(-1).detach().clamp(0,1)
        bg_weight = bg_valid.float()*(1-objectness)
        mean,covariance,effective = background_statistics(bg,bg_weight,self.shrinkage,self.ridge_ratio,self.ridge_floor)
        if self.background_override is not None:
            if self.training:
                raise ValueError("Background intervention is evaluation-only")
            mean = self.background_override["mean"].to(mean).expand_as(mean)
            covariance = self.background_override["covariance"].to(covariance).expand_as(covariance)
            effective = self.background_override["effective_support"].to(effective).expand_as(effective)
        fallback = effective<self.min_background_support
        diagonal = torch.diag_embed(covariance.diagonal(dim1=-2,dim2=-1))
        used_cov = diagonal if self.mode=="diagonal" else torch.where(fallback[...,None,None],diagonal,covariance)
        q = self.query_norm(queries.float())
        prototype = F.normalize(self.target(q),dim=-1,eps=1e-6)
        direction = prototype-mean
        w = torch.linalg.solve(used_cov,direction.unsqueeze(-1)).squeeze(-1)
        denominator = (direction*w).sum(-1).clamp_min(1e-8).sqrt()
        score = ((z-mean.unsqueeze(-2))*w.unsqueeze(-2)).sum(-1)/denominator.unsqueeze(-1)
        scale = .25+3.75*self.compatibility_scale(q).sigmoid()
        distance = (((z-prototype.unsqueeze(-2))*scale.unsqueeze(-2)).square()).mean(-1)
        compatibility = torch.exp(-.5*distance)
        contrast = (score/self.temperature).sigmoid()
        gate = compatibility*contrast
        if self.mode=="no_compatibility":
            gate = contrast+0*compatibility
        elif self.mode=="plain":
            gate = torch.ones_like(gate)+0*gate
        if self.mode!="plain":
            gate = gate*(effective>=2).unsqueeze(-1)
        logits = (z*self.semantic_query(q).unsqueeze(-2)).sum(-1)/math.sqrt(self.spectral_dim)
        attention = masked_softmax(logits,valid)
        contribution = attention*gate
        pooled = (contribution.unsqueeze(-1)*self.value(z)).sum(-2)
        residual = self.residual_scale*self.readout(pooled)
        record = None
        if context["diagnostics"] or context.get("train_summary"):
            record = {"active":torch.ones_like(effective,dtype=torch.bool),"candidate_points":points,
                "background_points":bg_points,"candidate_valid":valid,"background_weights":bg_weight,
                "background_mean":mean,"background_covariance":used_cov,"effective_background_support":effective,
                "diagonal_fallback":fallback,"target_prototype":prototype,"compatibility_scale":scale,
                "discriminant_direction":w,"background_score":score,"object_compatibility":compatibility,
                "evidence_gate":gate,"semantic_attention":attention,"aggregation_contribution":contribution,
                "residual_norm":residual.norm(dim=-1),"input_reference":references.detach()}
            if context["diagnostics"]:
                record.update(candidate_embeddings=z, background_embeddings=bg,
                    query_before=queries, query_after=queries+residual,
                    semantic_query=self.semantic_query(q))
        return residual,record

    def forward(self, queries, references, context, layer_index):
        if queries.shape[:2]!=references.shape[:2] or references.shape[-1]!=5 or queries.shape[-1]!=self.hidden_dim:
            raise ValueError("Spectral query/reference shape mismatch")
        context["visited_layers"] = max(context["visited_layers"],layer_index+1)
        if layer_index!=self.apply_layer:
            return torch.zeros_like(queries)
        if "encoder_objectness" not in context:
            raise ValueError("Spectral evidence requires encoder prediction context")
        residuals,records,summaries = [],[],[]
        # Covariance solves, small-value divisions and compatibility evaluation
        # remain FP32 even when the surrounding detector uses autocast.
        with torch.autocast(device_type=queries.device.type,enabled=False):
            for start in range(0,queries.shape[1],self.query_chunk_size):
                residual,record = self._chunk(queries[:,start:start+self.query_chunk_size],references[:,start:start+self.query_chunk_size],context)
                residuals.append(residual)
                if record is not None and context["diagnostics"]:
                    records.append({k:v.detach() for k,v in record.items()})
                if record is not None and context.get("train_summary"):
                    valid = record["candidate_valid"].detach()
                    n = valid.sum(-1).clamp_min(1)
                    summaries.append({
                        "gate_mean": (record["evidence_gate"].detach()*valid).sum(-1)/n,
                        "compatibility_mean": (record["object_compatibility"].detach()*valid).sum(-1)/n,
                        "gate_below_001_fraction": ((record["evidence_gate"].detach()<.01)*valid).sum(-1)/n,
                        "contribution_mass": record["aggregation_contribution"].detach().sum(-1),
                        "valid_fraction": valid.float().mean(-1),
                        "background_support": record["effective_background_support"].detach(),
                        "diagonal_fallback_fraction": record["diagonal_fallback"].detach().float(),
                        "residual_norm": record["residual_norm"].detach(),
                    })
        if records:
            context["records"][layer_index] = {k:torch.cat([r[k] for r in records],dim=1) for k in records[0]}
        if summaries:
            combined = {k:torch.cat([r[k] for r in summaries],dim=1) for k in summaries[0]}
            ordinary = context.get("ordinary_queries", queries.shape[1])
            split = queries.shape[1]-ordinary
            context["train_statistics"] = {"schema_version":"spectral-train-v1", "mode":self.mode,
                "apply_layer":self.apply_layer, "groups":{}}
            for name, section in (("ordinary",slice(split,None)),("denoising",slice(0,split))):
                if not combined["gate_mean"][:,section].numel():continue
                context["train_statistics"]["groups"][name] = {
                    k:{"mean":v[:,section].mean(),"min":v[:,section].min(),"max":v[:,section].max()}
                    for k,v in combined.items()}
        return torch.cat(residuals,dim=1).to(queries.dtype)

    def diagnostics(self, context):
        result = {"schema_version":"spectral-evidence-v1","mode":self.mode,
                  "coordinate_space":"normalized_canvas_xy","apply_layer":self.apply_layer,
                  "background_intervention":self.background_override is not None}
        records = context["records"]
        if records:
            active = records[self.apply_layer]
            result.update({k:torch.stack([records[i][k] if i in records else torch.zeros_like(v)
                for i in range(context["visited_layers"])]) for k,v in active.items()})
        return result
