# CODrone D-FINE-S OBB baseline

## Coordinate convention

CODrone annotation files use DOTA quadrilaterals: four consecutive vertices,
class name, and an optional difficulty flag. The data layer converts each
rectangle to `[cx, cy, w, h, theta]` with these invariants:

- pixel-space `theta` is in `[0, pi)` and positive clockwise in image coordinates;
- the long edge is always `w`, so `w >= h`;
- model targets are `[cx/W, cy/H, w/W, h/H, theta/pi]`;
- evaluator and exported predictions are restored to original-image pixels.

Images are resized with preserved aspect ratio and padded only on the right and
bottom. `size`, `scale_factor`, and `padding` are retained in every target so
postprocessing exactly reverses this operation.

## Training

The full config is `configs/dfine/dfine_hgnetv2_s_codrone_obb.yml`. The tiny
pipeline config is `configs/dfine/dfine_hgnetv2_s_codrone_obb_smoke.yml`.
Neither config uses Mosaic, MixUp, or batch multi-scale resizing.

CODrone evaluation follows DOTA/MMRotate-style OBB AP at IoU 0.5 and 0.75.
The training `best` checkpoint uses `mAP50/75 DOTA-07`, the mean of
`AP50_DOTA07` and `AP75_DOTA07`; COCO-style `mAP@[.50:.95]` is kept only as a
diagnostic number.

```bash
conda run -n wyq-deim python train.py \
  -c configs/dfine/dfine_hgnetv2_s_codrone_obb.yml -d cuda
```

D-FINE keeps fine-grained distribution refinement for `(cx, cy, w, h)` and
adds a periodic iterative angle branch. Matching uses class, L1, periodic angle,
KLD, and corner Chamfer costs. Training uses FDR localization, periodic angle,
and differentiable KLD geometry losses. Rotated IoU is used for quality and
evaluation; class-aware rotated NMS remains enabled for this baseline.

## Inference

```bash
conda run -n wyq-deim python tools/inference/obb_infer.py \
  --config configs/dfine/dfine_hgnetv2_s_codrone_obb.yml \
  --checkpoint logs/dfine_hgnetv2_s_codrone_obb/best_stg1.pth \
  --input /path/to/images --output ./obb_predictions
```

The command writes rendered images and per-image DOTA-style text predictions.
