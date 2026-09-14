# MODA：FressDet 论文设置配方

当前主实验入口是
[`dfine_obb_o2_fressdet.yml`](../../../configs/experiments/moda/dfine_obb_o2_fressdet.yml)。
所有方法与消融共用
[`fressdet_paper_protocol.yml`](../../../configs/experiments/moda/fressdet_paper_protocol.yml)。
设置以用户提供的 FressDet v2 论文为优先，评估实现对照其发布代码。
本地来源和实现细节保存在[内部核对记录](fressdet_reference_audit.md)。
论文主表按常规给已有方法引用文献，不增加结果来源标记列；实验设置描述实际运行配方。

## 主配方

| 项目 | 设置 |
|---|---|
| 数据 | 官方 train 9,156 图、test 4,885 图；八波段、八类 |
| 输入 | 原始宽 1,200 × 高 900，右侧/底部补零至宽 1,216 × 高 928 |
| 训练 | 20 epochs；全局 batch 8，两卡各 4；保留最后不足一批的样本 |
| 初始化与增强 | 从头训练；无随机增强；全 backbone 可训练 |
| 优化器 | AdamW，LR 0.01、betas=(0.937, 0.999)、weight decay 0.0005 |
| 学习率 | 线性衰减，终点比例 0.01；前三轮 warmup，bias 起始 LR 0.1，其余为 0 |
| 精度与 EMA | FP32；EMA decay 0.9999、warmups 2,000；梯度裁剪 10 |
| O² 后处理 | top-300，无 NMS；score > 0.01；每图最多 300 框 |
| 主指标 | ProbIoU 匹配，AP50、AP75、mAP@[.50:.95]；101 点插值后梯形积分 |
| 评估与保存 | 每轮评估；按 AP50 保存 `best_stg1.pth`，另存 `last.pth` |
| 随机种子 | 首组 seed 0；配对实验使用相同种子 |

这个 benchmark 入口的每轮评估使用官方 test，不再额外从 train 划走开发图。
正式运行统一使用此配方；代码检查通过显式 debug 清单覆盖，不另维护一套训练配置。
本项目保持 O² 的 ADR、DN、matcher、loss 与 500 个普通 queries；不把 YOLO 的
box/cls/DFL 损失系数套进不同结构的 criterion。

矩形画布下，四个长度/中心坐标统一除以 `S=max(W,H)=1216`，角度仍除以 π。
decoder anchors、旋转采样、光谱分支取样、原图框恢复共同使用该坐标约定，避免分别除以宽高
后改变旋转矩形几何。默认旧配置仍使用原坐标模式。

`--geometric` 可额外计算同一批预测的几何旋转 IoU 指标，字段前缀为 `riou_`，供定位分析使用；日常评估不重复计算这组指标。
这些字段采用同样的匹配与积分方式。通用 `DotaOBBEvaluator` 仍供其他数据集的
DOTA 协议使用；MODA 主配方的 checkpoint 选择指标明确为 AP50。

## 基线、创新与双 3090 运行

所有命令统一维护在 [实验运行指南](RUN_EXPERIMENTS.md)：基线、普通光谱聚合、嵌入通道对角统计、去对象约束和完整方法各有独立入口；支持预检、训练、续训、评估和源像素可视化。

```bash
conda activate wyq-deim
python tools/experiments/moda.py list
python tools/experiments/moda.py train baseline --gpus 0,1 --seed 0
python tools/experiments/moda.py train full --gpus 0,1 --seed 0
```

正式运行前在目标机器执行指南中的 `preflight --full-size --batch-size 4 --dense`。这条预检保留论文配方的原图 resize/pad 顺序；CPU 小画布预检仅检查代码。双 3090 峰值显存和完整训练耗时仍需目标机器实测。

## 已完成验证

本地 FressDet 函数与本实现的 ProbIoU、逐阈值匹配、AP 积分及 fast NMS 已做数值
对照。矩形坐标的几何保持、原图恢复、旋转 attention 和两种 decoder 的训练反向
通过测试。真实 MODA 小样本检查仅验证代码运行，不作为正式检测精度结果。
