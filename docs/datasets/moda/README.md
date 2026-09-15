# MODA 基线

当前只有八波段 `baseline` 和伪 RGB `rgb`，尚未确定创新方案。
在项目根目录、`wyq-deim` 环境运行：

```bash
python tools/experiments/moda.py train rgb
python tools/experiments/moda.py eval rgb
```

八波段将 `rgb` 换成 `baseline`。默认 GPU 0,1、20 轮、batch 8、FP32；用 `--gpus` 指定卡。
训练自动评估并保存指标和诊断；验证默认用最佳权重。`--checkpoint 路径` 可指定验证权重或续训。
结果位于 `logs/moda/experiments/<baseline或rgb>/seed0_fp32/`，报告各类 AP50 和整体 mAP50/mAP75/mAP（ProbIoU）。
伪 RGB 保留 B4/B2/B1，现有八输入模型的其余五路在训练和验证时恒为零。
