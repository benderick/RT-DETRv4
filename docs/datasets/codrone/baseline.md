# CODrone OBB 工程底座

## 坐标约定

CODrone 的 DOTA 四点标注会转换为 `[cx, cy, w, h, theta]`。像素角度采用图像坐标系下的顺时针方向，并规范化到 `[0, pi)`；长边总是 `w`，因此 `w >= h`。模型 target 归一化为 `[cx/W, cy/H, w/W, h/H, theta/pi]`。后处理会把每个预测恢复到原图像素坐标。

图像保持长宽比，只在右侧和底部 padding。`size`、`scale_factor` 和 `padding` 会保留在每个 target 中，使推理能够精确反变换。

## 模型入口

O² 是主 OBB 实现。唯一保留的对照是独立的标量角度回归头。两个架构各有一个
原图配置和一个标准切片配置；精确的四入口约定记录在
[框架 OBB 约定](../../framework/obb_conventions.md)。

两条路径都不使用 Mosaic、MixUp 或 batch multi-scale resize。当前正式 recipe 使用 1024 网络画布、30 epoch，以及相同的 HGNetv2-B2 模型容量，用于受控比较。

CODrone 评估报告 IoU 0.5 和 0.75 下的 DOTA-07 AP。checkpoint 选择使用 `mAP50/75 DOTA-07`；COCO 风格的 `mAP@[.50:.95]` 仅作为诊断指标。

```bash
conda run -n wyq-deim python train.py \
  -c configs/dfine/dfine_obb_o2.yml -d cuda --seed 0
```

## 结构化实验证据

四个配置都启用诊断日志。一次运行会在 `output_dir/diagnostics` 下写入 manifest、采样训练记录和完整验证证据。schema 包括：

- 原图像素坐标下的 GT 与预测几何；
- Hungarian query 身份，以及类别、中心、角度、尺度、Chamfer、KLD 和 rotated-IoU 证据；
- 每个 score/NMS/max-detection 决策，以及造成抑制的候选框；
- decoder layer 轨迹和 O² 分布；
- 采样图像上实际应用的增强参数；
- loss 项、学习率、step 耗时、GPU 显存和模块梯度范数；
- tile sidecar 可用时的 source object、tile origin、IOF、重复次数和边界距离。

记录以 gzip JSONL stream 形式按 rank 原子写入，并在 solver 的 failure-safe cleanup 路径中关闭。因此，分析直接从实验目录读取；不需要用户手动整理数据。

```bash
conda run -n wyq-deim python tools/analysis/summarize_obb_diagnostics.py \
  logs/dfine_obb_o2 --output ./obb_analysis

conda run -n wyq-deim python tools/analysis/visualize_obb_mechanisms.py \
  logs/dfine_obb_o2 --output ./obb_mechanism_cases

conda run -n wyq-deim python tools/analysis/compare_obb_mechanisms.py \
  logs/dfine_obb_angle logs/dfine_obb_o2 --output ./obb_paired_mechanisms
```

成对分析工具通过 `image_name + gt_index` 连接同一个目标，使用共享 crop，并按选定的机制级改进排序，而不是只展示最终框更好看的例子。

## 推理

```bash
python tools/inference/obb_infer.py \
  --config configs/dfine/dfine_obb_o2.yml \
  --checkpoint logs/dfine_obb_o2/best_stg1.pth \
  --input /path/to/images --output ./obb_predictions \
  --score-threshold 0.3
```

标准 `1180/200` 切片协议和合并到原图后的推理流程记录在
[切片协议](tiling.md)。
