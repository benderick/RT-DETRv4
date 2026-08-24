# 旋转检测框架集成契约

本契约约束数据集、模型、损失、评估器、日志和测试的接入方式。目标是让框架持续
扩展时仍能保证 direct-angle 与 O² 两条稳定链路可训练、可推理、可评估。

## 1. 当前公共边界

稳定模型只允许两个 `refinement_mode`：

| 值 | 几何输出 | 局部分布 |
|---|---|---|
| `direct_angle` | FDR `cxcywh` + 标量周期角度 | 4 个 FDR 分布 |
| `o2_adr` | O² ADR 解码 OBB | 6 个 ADR 分布 |

稳定模型定义只有：

```text
configs/dfine/dfine_obb_angle.yml
configs/dfine/dfine_obb_o2.yml
configs/dfine/dfine_obb_angle_tile.yml
configs/dfine/dfine_obb_o2_tile.yml
```

这些配置名描述模型差异，不编码数据集名、backbone 尺寸、tile 尺寸或输入分辨率。
`_tile` 配置复用同一模型，只替换数据和合并求值协议。新增数据集不复制模型实现；
完整可运行的数据集绑定放在 `configs/experiments/<dataset>/`，通过 include 组合稳定
模型定义与 `configs/dataset/` 中的数据协议。

## 2. 目录职责

```text
engine/data/dataset/              # 数据集 adapter 与通用 DOTA OBB evaluator
engine/data/transforms/           # 方法无关的 OBB 数据增强
engine/evaluation/                # 指标与原图/tile 合并评估
engine/diagnostics/               # 方法无关的结构化日志
engine/rtv4/                      # detector、matcher、criterion、postprocessor
engine/rtv4/obb/methods/o2/       # O² 私有 ADR 原语

configs/dataset/                  # 数据协议
configs/dfine/                    # 稳定训练入口
configs/experiments/<dataset>/    # 数据集与稳定模型的可运行绑定
docs/framework/                   # 公共契约与坐标规范
docs/datasets/<name>/             # 数据集说明
docs/research/o2/                 # O² 复现资料与审计
tools/inference/                  # 通用推理入口
tools/analysis/                   # 只读日志分析与可视化
tools/research/o2/                # O² 私有分析工具
test/framework/                   # 公共底座测试
test/research/o2/                 # O² 私有测试
```

方法私有公式不得塞入通用 evaluator、dataset adapter 或 postprocessor。可以复用的
几何与评估能力不得以数据集名复制注册。

## 3. OBB 坐标契约

模型内部统一使用：

```text
[cx/W, cy/H, w/W, h/H, theta/pi]
w >= h, theta in [0, 1)
```

图像空间日志和求值统一使用像素中心、像素边长及弧度角。所有入口必须显式处理：

- quadrilateral 与 rotated box 的转换；
- resize、pad、flip、rotation 后的合法化；
- tile 坐标到原图坐标的恢复；
- 长边规范和半周角周期；
- 空标注、difficulty/ignore 和越界框。

详细定义见 `docs/framework/obb_conventions.md`。

## 4. 新数据集

新增数据集必须提供：

1. adapter，输出统一的 `boxes/labels/image_id/orig_size/size`；
2. 类别表、标注来源和 split 清单；
3. 空图、非法多边形和 difficulty/ignore 的明确语义；
4. 数据 provenance，包括 root、split 和 inventory hash；
5. 数据单元测试与 perfect-prediction evaluator 测试。

DOTA 兼容数据集必须复用 `DotaOBBEvaluator`；切片求值复用
`MergedDotaOBBEvaluator`。禁止创建 `<DatasetName>Evaluator` 来复制同一指标逻辑。

## 5. 新模型、损失和模块

新增模型能力必须拥有一个 canonical Python 路径和一个明确配置选择器。不得同时
保留布尔别名、旧类名和多个等价注册名。

接入前必须满足：

- 不改变 direct-angle/O² checkpoint 的参数名和张量形状；
- 不改变二者默认训练和推理图；
- regular、aux、encoder、pre、DN 输出契约完整；
- CPU forward/backward、空 GT 和 eval forward 通过；
- AMP 下不靠静默 `nan_to_num` 或局部强制 FP32 掩盖非有限梯度；
- 新 loss 的权重、匹配成本和归一化语义在配置中显式可审计。

如果确需改变公共 checkpoint 或日志格式，必须提升版本并提供迁移说明，不能保留
无人使用的兼容层。

## 6. 数据增强

增强必须同时变换图像和 OBB，并在输出前执行统一合法化。每种随机增强必须将实际
参数写入 target 元数据，至少覆盖：

- resize 比例与 padding；
- horizontal/vertical/diagonal flip；
- rotation 角度；
- photometric 顺序和参数。

训练配置不启用 Mosaic 和 MixUp。新增增强需要固定随机种子测试、几何 round-trip
测试、有限值测试和可视化抽查。

## 7. 评估与后处理

正式指标在原图坐标计算。普通 full-image 和 tile 推理必须显式区分后处理语义：

- O² full-image 是 DETR 集合预测，默认不做 overlap NMS；
- direct-angle full-image 暂时保留已固化基线的历史 NMS 语义；
- tile 模式保留局部 NMS，并在恢复到原图后执行全局类别感知 rotated NMS。

`RotatedPostProcessor.apply_nms` 必须由稳定配置解析为确定值。诊断对照可以在同一次
冻结输出上按调用覆盖，但不能把 NMS-free 与 NMS 的指标混报。tile 评估器必须检查
tile 完整性，不能用缺失 tile 的结果静默计算 AP。

至少报告：

- DOTA-07 AP50/AP75 与主汇总；
- diagnostic mAP@[.50:.95]；
- per-class AP；
- 候选数、NMS 抑制关系和最终检测来源。

## 8. 诊断日志

实验由使用者运行，分析端只读取日志。日志必须直接保存以后可能用于论文分析的过程
证据，不要求使用者重新整理数据。

每个 run 至少记录：

```text
schema_version / run_id / config_hash / git state
dataset / split / inventory hash / checkpoint / epoch / step / seed
refinement_mode / world_size / device / dtype
```

训练记录至少包括 loss 分项、匹配、中心/尺度/角度/rIoU、分布统计、梯度、AMP scale
与 skipped step、数据等待时间、step time 和 peak memory。推理记录至少包括预处理、
backbone、encoder、decoder、postprocess/NMS 时间以及峰值显存。

逐层 query 记录必须保留 `reference_box`、预测框、类别置信度和方法对应的局部分布；
O² 还需记录 ADR offset、codebook 覆盖和坐标缝暴露。大张量放 artifact，summary 只存
索引与 hash。

## 9. 测试门槛

### 数学与几何

round-trip、周期等价、近轴/近方形、极端尺寸、非法输入、有限梯度和 CPU/CUDA
（可用时）。

### 数据与求值

原图/tiny/tile 读取、增强确定性、IOF、padding、空图、perfect prediction、Task1
导出、原图恢复、tile 完整性和 rotated NMS。

### 模型链路

direct-angle 与 O² 均测试训练 forward、全部 loss、backward、eval forward、后处理、
可视化和诊断落盘；已有 checkpoint 必须严格加载。

### 公共入口

四个稳定配置可解析；只注册通用 evaluator；构造器拒绝除 `direct_angle` 和 `o2_adr`
以外的 refinement mode。

统一测试入口：

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python test/run_all.py
```

CUDA 或真实数据不可用时可以 skip，但必须显示原因；断言失败不能改成 skip。

## 10. 提交检查表

- [ ] 目录位置符合职责，注册名唯一；
- [ ] 坐标、类别和 ignore 语义已写明；
- [ ] 配置不含机器专属绝对路径；
- [ ] direct-angle 与 O² 的构建、严格加载、训练和推理回归通过；
- [ ] 新增过程信息进入结构化日志；
- [ ] evaluator 与分析工具不复制已有逻辑；
- [ ] 测试位于 `test/`，文档链接无失效路径；
- [ ] 没有缓存、临时输出或实验日志进入源码目录。
