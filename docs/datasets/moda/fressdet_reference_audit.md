# 本地 FressDet v2 内部核对

日期：2026-09-12。用户提供的参考目录：
`research/references/FressDet/paper/arXiv-2607.05148v2/main.tex` 与
`research/references/FressDet/code/FressDet/`。外部资料不进入本项目 Git。
本文保存配方选择依据；运行说明见 [FressDet 论文设置配方](fressdet_alignment.md)。
论文主表采用常规方法文献引用，不要求额外标记列，也不将重训所有引用方法作为前置条件。

## 方法边界

v2 描述 SpeIW 连续、有序、单调的光谱重采样；ReCoW 的光谱 soft routing 与空间
hard routing 产生一致性驱动的残差调制；C4 旋转等变 backbone/neck 与 oriented-aware
head。v2 明确修正了早期附录的描述：ReCoW 不是两个分支的标量凸组合。
BCSR 研究预测框边界的局部光谱证据及逐层定位更新；不以“光谱注意力＋旋转”本身
作为新颖性，不把 C4 的条件等变性泛化为任意角度严格等变性。

## 论文优先的配置选择

| 项目 | FressDet 论文 / 发布代码 | 本项目主实验 |
|---|---|---|
| 数据 | 论文 MODA train/test 9,156/4,885；代码 YAML 为本机占位路径 | 原始官方 train/test，全训练集 |
| epochs | 两者均为 20 | 20 |
| 输入 | 论文宽 1,216 × 高 928；代码 imgsz=1200 经 stride 修正至 1216，rect=False | 采用论文矩形大小，原图右/下补零 |
| batch | 论文 8；示例 train.py 为 12 | 全局 8，两卡各 4 |
| 初始化/增强 | 论文从头训练、无增强；代码 YAML 构建模型不加载权重，增强系数均为 0 | 从头训练、无随机增强 |
| 优化 | AdamW，LR 0.01，线性终点 0.0001，WD 0.0005，warmup 3 epochs | 同名设置；保留 O² 原生 criterion |
| 梯度累积 | 代码 nbs=64，batch8 时稳态累积 8 步，warmup 从 1 步增长；论文未单列 | 每个全局 batch8 更新，不移植 YOLO 的 nbs 累积策略 |
| 评估阈值 | validator 未指定 conf 时 OBB 默认 0.01；NMS IoU 0.7，max_det 300 | 相同阈值与上限，逐类 ProbIoU fast NMS |
| 框匹配 | 论文文字描述 rotated-rectangle IoU；`obb/val.py` 实际调用 batch_probiou | 以发布代码 ProbIoU 作为主匹配，几何 IoU 另存诊断 |
| AP | 101 个横坐标插值，再 trapz 积分；阈值来自 float32 linspace(.5,.95,10) | 数值对照通过，协议名固定 |
| 最佳模型 | `utils/metrics.py: Metric.fitness` 权重 `[0,0,1,0]`，即 AP50 | 每轮按 AP50 保存 EMA checkpoint |
| 精度 | 发布配置 AMP=False | FP32；可显式覆盖且配对实验一致 |

warmup 的 3 轮按 `3*ceil(9156/8)=3435` 次迭代计算；bias 从 0.1 起，其他组从 0 起，
插值目标随当前 epoch 的线性 LR 更新。AdamW 使用 beta1=0.937，代码只对含
`momentum` 字段的优化器执行 momentum warmup，因而不把 beta1 改成 0.8。
线性调度采用发布实现的 epoch/20：第 20 轮实际 LR 为 0.000595，轮后调度边界达到
0.0001。归一化和 bias 参数不施加 weight decay。

这里对齐实验预算和评估实现，并非把 O² 改写为 FressDet 的 YOLO 训练器。
O² 的 backbone、500 queries、DN、ADR、损失权重及参数量属于模型自身设置。
矩形 padding 的位置和颜色在论文中未细述，本项目明确选右/下补零，并同步 valid_mask。

## 指标实现与类别

`BenchmarkOBBEvaluator` 使用协议
`ultralytics_obb_probiou_interp101_trapz_v1`。差分测试直接从本地文件抽取
`batch_probiou`、`match_predictions`、`compute_ap`、`nms_rotated`，避免导入整个
FressDet 或新增依赖。匹配保留发布代码两次 `np.unique` 的排序语义；fast NMS
保留“已被抑制的高分框仍可抑制低分框”的行为。理想单目标 AP 为 0.995，来自该
插值积分边界约定，不改成 DOTA evaluator 的 1.0。

原始 MODA 类别顺序为 `car,van,truck,bus,tricycle,bike,awning-bike,pedestrian`。
FressDet 附带 `drmod.yaml` 顺序为
`car,pedestrian,bike,awning-bike,van,truck,bus,tricycle`。若未来读取遵循该 YAML
的预测，转换到原始 MODA ID 使用 `[0,7,5,6,1,2,3,4]`；按类别名称引用论文各类 AP。
目前本项目模型始终使用原始 MODA 顺序，不读取外部权重。

## 论文数值记录

本地 v2 的 MODA 主表：FressDet AP50/AP75/mAP 为 73.1/60.2/54.3，参数 2.3M；
OSSDet 为 69.0/45.9/42.7。README 的 73.65/54.51 不用于替换论文主表数字。
这些是参考论文报告值；本项目尚无完整 MODA 训练结果。
