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
| 后处理 | score > 0.01；逐类 ProbIoU fast NMS，阈值 0.7；每图最多 300 框 |
| 主指标 | ProbIoU 匹配，AP50、AP75、mAP@[.50:.95]；101 点插值后梯形积分 |
| 评估与保存 | 每轮评估；按 AP50 保存 `best_stg1.pth`，另存 `last.pth` |
| 随机种子 | 首组 seed 0；配对实验使用相同种子 |

这个 benchmark 入口的每轮评估使用官方 test，不再额外从 train 划走开发图。
用于调整结构和超参数的 train/dev 入口仍保留在旧配置中；两套运行使用不同日志目录。
本项目保持 O² 的 ADR、DN、matcher、loss 与 500 个普通 queries；不把 YOLO 的
box/cls/DFL 损失系数套进不同结构的 criterion。

矩形画布下，四个长度/中心坐标统一除以 `S=max(W,H)=1216`，角度仍除以 π。
decoder anchors、旋转采样、BCSR、原图框恢复共同使用该坐标约定，避免分别除以宽高
后改变旋转矩形几何。默认旧配置仍使用原坐标模式。

评估额外保存同一批预测的几何旋转 IoU 指标，字段前缀为 `riou_`，供定位分析使用。
这些字段采用同样的匹配与积分方式；旧 `DotaOBBEvaluator` 的 DOTA-07/dense AP
保留给旧配置。选择 checkpoint 的主指标在日志中明确记录为 AP50。

## 双 3090 运行

先补齐 `data/MODA/train/images`。当前本地只有 1,000 张训练图，完整配置会报缺图，
不会自动缩成已有图子集。无需安装新包。

```bash
conda activate wyq-deim
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py -c configs/experiments/moda/dfine_obb_o2_fressdet.yml --seed 0
```

BCSR 主方法使用同一配方：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py -c configs/incubator/bcsr/moda_edge_fressdet.yml --seed 0
```

整框共享路由和固定初始框消融分别使用 `moda_object_fressdet.yml`、
`moda_initial_fressdet.yml`。更多种子用独立 `--output-dir`。命令显式传 `--seed 0`，
因为通用训练 CLI 的默认 seed 42 会覆盖 YAML 字段。

复查最佳 checkpoint：

```bash
python train.py -c configs/incubator/bcsr/moda_edge_fressdet.yml --test-only -r logs/research/bcsr/moda_edge_fressdet/best_stg1.pth --seed 0 --output-dir logs/research/bcsr/moda_edge_fressdet_test
```

目标机器先测密集样本的 FP32 显存：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/dataset/moda_preflight.py --config configs/incubator/bcsr/moda_edge_fressdet.yml --device cuda:0 --height 928 --width 1216 --queries 500 --batch-size 4 --steps 4 --dense --output logs/research/bcsr/paper_protocol_dense_cuda.json
```

preflight 为检查指定画布，会将样本 resize 到该画布；正式配方使用原始大小再补边。
它覆盖密集 GT 的动态 DN 数量，但不包含 EMA、DDP 通信缓冲区或验证集全量求值的
总开销，正式运行还需短程测量。双 3090 的峰值显存和 12 小时完成性尚未验证。
如需 AMP，可在配对训练命令中共同加 `-u use_amp=True` 并使用新输出目录；显存不足时
先检查实际峰值，再决定调整，不在代码中静默降 batch 或输入尺寸。

## 已完成验证

本地 FressDet 函数与本实现的 ProbIoU、逐阈值匹配、AP 积分及 fast NMS 已做数值
对照。矩形坐标的几何保持、原图恢复、旋转 attention 和两种 decoder 的训练反向
通过测试。真实 MODA 小样本检查仅验证代码运行，不作为正式检测精度结果。
