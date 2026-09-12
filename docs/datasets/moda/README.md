# MODA 八波段旋转检测

当前主实验采用 [FressDet 论文设置配方](fressdet_alignment.md)：完整 train/test、
20 epochs、宽 1216 × 高 928、全局 batch8、从头训练，以及发布代码的 ProbIoU AP。
下文的 36 轮 train/dev 配方保留为开发入口，使用独立配置与日志目录。

适配器：[moda_dataset.py](../../../engine/data/dataset/moda_dataset.py)。稳定入口：
[O²](../../../configs/experiments/moda/dfine_obb_o2.yml)、
[direct-angle](../../../configs/experiments/moda/dfine_obb_angle.yml)。

## 数据与协议

目录为 `data/MODA/{train,test}/{images,labels}`。原始 `.npy` 是 uint8
`[8,W,H]`（本地为 `[8,1200,900]`），读取后交换空间轴成为 `[8,H,W]`。
八个通道保持源顺序，以 float/255 输入；不声称这八个索引已校准为具体波长。
所有通道同步 resize/flip/rotate/pad，禁用 RGB 专用色彩增强、Mosaic 和 MixUp。
`valid_mask` 同步变换，标出旋转空白和 padding。预览使用标注清楚的第 0 波段灰度图。

类别 ID：`car, van, truck, bus, tricycle, bike, awning-bike, pedestrian`。
源 difficulty=0/1/2 均作为有效目标，另存 `source_difficulty`，不修改源标注。
这对应 MODA 官方 loader 的默认 difficulty 阈值 100；不能直接沿用普通 DOTA 的
“非零即忽略”。源标注与 split 都有 SHA256 记录。

按完整训练源 9,156 张、测试集 4,885 张配置。2026-09-12 本地训练图仅有 1,000 张，
标签完整；缺图会明确报错，不自动缩成可用子集。全部训练标签含 239,859 个目标，
测试标签含 90,323 个目标。现有 1,000 张是文件排序前缀，小目标比例偏低，不能用于
判断完整数据上的方法优劣。

## 分割

MODA 没有独立官方开发集。以下开发入口从 train 划分 train/dev，test 用于最终评估；
上面的 FressDet benchmark 入口采用完整官方 train/test。
推荐提供 JSON `{image_stem: verified_scene_group}`，覆盖所有训练标签：

```bash
conda activate wyq-deim
python tools/dataset/prepare_moda_splits.py --output logs/moda/splits --group-map /path/to/scene_groups.json
```

没有场景表时可显式选用 `--date-prefix-proxy` 替代 `--group-map`。本地已生成该临时
协议：train=7,645、dev=1,511，双方都含八类，日期前缀不交叉。**日期不是已验证的
场景 ID**，不能声称该分割已排除邻帧/同场景泄漏；论文实验前应核实真实场景映射。
清单输出目录存在时拒绝覆盖，需要换新目录并更新两个 dataset.split_file。

为代码检查显式生成可用图子集（本地已经存在 16 图清单，无需重复执行）：

```bash
python tools/dataset/prepare_moda_splits.py --output logs/moda/debug_split --debug-images 16
python tools/dataset/moda_preflight.py --debug-split logs/moda/debug_split/debug.json
```

预检查默认 CPU、128 尺度、64 queries、真实标注、三次优化更新，覆盖完整 loss/DN、
反向和推理。它不能提供正式 AP、1024 显存或训练耗时估计。

## 双 3090 训练入口

先补齐数据、确认 split，运行目标分辨率的最密集样本显存检查：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/dataset/moda_preflight.py --device cuda:0 --amp --size 1024 --queries 500 --batch-size 4 --dense --output logs/moda/preflight_1024_dense.json
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py -c configs/experiments/moda/dfine_obb_o2.yml --seed 42
```

默认全局 batch=8，即每卡 4，AMP、1024 输入、500 普通 queries、36 epochs；每 3 epoch
评估，12 epoch 保存较完整诊断。实际双 3090 显存和 12 小时预算尚未测量；必须据
上述检查和短程吞吐测量调整。O² 保留 `released_dynamic` DN，密集样本会增加 DN
数量，不应只测少目标样本。若要降 batch，对所有配对实验同步调整并重跑基线。

HGNetv2 B2 的 RGB 预训练首层扩展为 8 通道：`mean_RGB(weight) * 3/8` 后复制。
八波段输入相同时与原 RGB 相同输入的卷积响应一致。权重沿用
`pretrain/hgnetv2/PPHGNetV2_B2_stage1.pth`，不自动安装包。

这里继承当前仓库 `7b5f727` 的调参：base/backbone LR=`2e-4/2e-5`，O²
`loss_angle=0`。它们与冻结 tag `obb-o2-baseline-v1` 的历史验收配方有三项差异；
历史 source-aligned 验收工具仍会如实报告 FAIL。MODA 配对对照必须共同使用当前
明确配方，不能称为已经通过历史训练配方验收。

开发协议冻结后，使用 `dfine_obb_o2_fulltrain.yml` 以全部 9,156 张 train 重训。
该入口显式设置 `eval_during_training: False`，不构造验证集或 evaluator，不据 test
选轮次；固定训练预算结束后使用 `last.pth`。例如将上面的训练配置替换为
`configs/experiments/moda/dfine_obb_o2_fulltrain.yml`。冻结 checkpoint 后测试：

```bash
python train.py -c configs/experiments/moda/dfine_obb_o2_test.yml --test-only -r /path/to/checkpoint.pth --output-dir logs/moda/final_test
```

`*_test.yml` 是测试专用入口，使用时必须带 `--test-only`；不用于选模型或调参数。

## 指标

复用 `DotaOBBEvaluator`：像素坐标旋转框几何 IoU，保留 DOTA-07 AP50/AP75，同时记录
101 点插值的 AP@[.50:.95]。本数据配置显式以 `selection_metric: mAP50_95` 选最佳
checkpoint；不改变旧配置默认的 `(AP50_07+AP75_07)/2`。
full-image O² 默认无 NMS，上限 500 预测，阈值 0.001。该 dense AP 不能未经验证称为
完全等同官方 COCO/MMRotate/Ultralytics。外部方法对比前见
[FressDet 源码核对](fressdet_reference_audit.md)。

## 通用扩展接口

`RotatedDFINETransformer.geometry_adapter` 默认 `None`，不增加旧参数树的键。
启用时 adapter 实现 `build_context(images, targets, diagnostics=bool)`、
`forward(queries, detached_current_boxes, context, layer_index)` 与 `diagnostics(context)`。
返回 `[B,Q,D]` 残差只进入几何头；原始分类输入、固定 ADR anchor、DN 和 matcher
语义保留。几何改变仍会通过 LQE 和下一层引用框间接影响分类，不能声称分类完全独立。
上下文限定在单次 forward 内，推理只允许用图像和有效区域元数据，不读 GT。

隔离模块可通过配置 `imports: [engine.…]` 注册；稳定模块不导入任何 idea。
零残差测试覆盖 direct-angle/O² 的输出、全部 loss 和梯度精确等价及 strict loading。
