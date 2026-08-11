# CODrone OBB baseline verification

Verified on 2026-08-11 with the `wyq-deim` conda environment and one NVIDIA
A100 GPU.

## Dataset audit

| split | images | parsed objects |
|---|---:|---:|
| train | 5,002 | 219,527 |
| val | 2,000 | 89,468 |
| test | 3,002 | 134,597 |
| total | 10,004 | 443,592 |

Every full-resolution JPEG passed Pillow verification. All images are
`3840x2160`; all annotations parsed to finite canonical OBBs with valid class
indices. Missing annotations in the tiny validation/test splits are returned as
empty targets. Explicit `ignored` regions in those splits are excluded from
training and respected by evaluation.

## Automated verification

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n wyq-deim python test/run_all.py
```

Result: 18/18 tests passed. Coverage includes geometry round trips, periodic
angles, rotated IoU/NMS, differentiable geometry loss, deterministic and random
augmentations, crowded-image denoising bounds, empty targets, matching/loss
backpropagation, original-coordinate restoration, perfect-prediction DOTA AP,
DOTA export, visualization, full dataset parsing, a fixed-feature convergence
test, and a real `1024x1024` D-FINE-S forward/backward/inference step.

The full-model step produced `600x5` OBB predictions with finite gradients. A
single-image run used about 1.49 GiB peak allocated memory; the batch-two
`train.py` smoke run used about 3.2 GiB.

## Trainer and inference smoke run

The two-epoch `train_t -> val_t` smoke configuration completed 16 optimizer
steps, both validation passes, TensorBoard logging, and checkpoint creation.
Mean total training loss changed from `27.3997` to `27.0609`. The second epoch's
first batch was `25.7939`, down from `26.4338` at the start. AP remains zero in
this deliberately short run because it covers only the first 16 steps of a
500-step warmup; it is a pipeline check, not a reported accuracy experiment.

The independent inference command successfully reloaded `last.pth`, ran rotated
NMS, restored original 4K coordinates, and wrote both a rendered JPEG and DOTA
text predictions. Formal accuracy should be reported only after the full
72-epoch configuration completes.
