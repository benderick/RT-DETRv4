# O²-DFINE 复现逻辑审计

> 结论：当前 `refinement_mode: o2_adr` 是可运行、可测试的 **paper-derived
> O²-DFINE reproduction**，不是作者官方 O²-DFINE 源码的逐行复刻。作者公开
> 仓库提供了 O²-RTDETR，而没有提供同等完整的 DFINE/ADR 实现；本项目因此把
> 论文公式、公开 O² 代码语义和本地必要补全逐项标记，不能混称“完全一致”。

2026-08-21 的失败训练根因、AMP 冻结证据和修复见
[`failed_run_audit_20260821.md`](failed_run_audit_20260821.md)。

## 1. 四个机制是否真的接入

| O² 机制 | 当前实现 | 验证入口 | 状态 |
|---|---|---|---|
| Angle Distribution Refinement | 外接 HBox 四边距离 + 两个滑动顶点偏移，共六个非均匀分布，逐层累计精炼 | `obb/methods/o2/adr.py`、`rotated_dfine_decoder.py` | 已接入 |
| Chamfer matching cost | 四角点集合 Chamfer，主配置按 O²-DFINE Eq.10 使用 squared distance | `rotated_box_ops.py`、`rotated_matcher.py` | 已接入 |
| Oriented Contrastive Denoising | box/angle/probability 三种 OBB 噪声原语，主配置使用论文/公开配置对应的 box mode | `rotated_denoising.py` | 已接入 |
| Rotated cross-attention | 5D reference 的 sampling offsets 在每层按当前 OBB 角度旋转 | `dfine_decoder.py` | 已接入 |

此外，criterion 会计算 decoder 层间 query-index instability；它是验证 OCD 动机
的诊断量，不应被单列成第五个模型模块。

## 2. ADR 复现链路

当前 O² head 不再是“D-FINE `xywh` 分布 + 标量角度”。它以初始 OBB 为锚，
编码外接水平矩形四侧距离和两个 gliding offsets：

```text
reference OBB
  -> external left/top/right/bottom + epsilon/eta
  -> 6 × (reg_max + 1) residual logits
  -> O² non-uniform codebook expectation
  -> six-value geometric decode
  -> legal OBB
```

主配置为 `reg_max=32, a=.5, c=.25`，codebook 端点、中心零值、单调性和插值
重建均有单元测试。六分量共用 D-FINE 的累计 logits 语义，Location Quality
Estimator 也从四分量扩为六分量。

### 论文未定义而工程必须回答的部分

六个独立分布的任意组合未必严格对应正交矩形，可能带 shear。论文没有完整
规定非法中间六值的处理。本项目只构造论文定义的 ordered
top/right/bottom/left 顶点，再使用框架统一的 ordered-quadrilateral 转 OBB：
中心取均值、两条相邻边给出边长、较长边给出方向。合法矩形是严格固定点。

此前借用 ai4rs `delta_midpointoffset_rbbox_coder.py` 的等对角径向投影属于错误的
证据外推，已删除；该 coder 最终依赖 CPU/OpenCV `minAreaRect`，不是公开的可微
O²-DFINE ADR。ADR 也不再隐藏强制 FP32 路径或 scaler 下限。真实根因和压力测试
见 [`failed_run_audit_20260821.md`](failed_run_audit_20260821.md)。

ADR 的编码还做了两项几何正确性修复：

- 顶点选择使用中心相对坐标和一致的轴吸附，避免 `torch.isclose` 容差随绝对
  图像坐标改变；
- 左右、上下使用联合 span，保留中心平移，不独立截断负边距。

这些修复确保平移不改变编码、合法框 round-trip、近轴框和 exact-square 有稳定
行为。

## 3. “Square Problem” 的审计结论

质疑中提到

\[
(w,h,\theta)\equiv(h,w,\theta+\pi/2)
\]

会让 top/right vertex 交换，进而产生两套 ADR。经过中心相对编码和确定性
canonicalization 后，这两组参数会得到同一几何 ADR 结果。因此同一个已标注
几何框不会仅因 `(w,h)` 交换而产生两套 target。

但 ADR 仍有更一般的 **chart seam**：物理框连续穿过图像坐标轴时，top/right
极值顶点身份换支，`epsilon/Wr`、`eta/Hr` 可从接近 0 跳到接近 1。这发生在普通
长方形上，不只发生在正方形。它是坐标图身份造成的拓扑缝，不能靠 epsilon 或
正方形 special case 根治。日志已记录 seam distance、offset fraction、残差幅度
和 codebook overflow，避免把几何连续误报为坐标连续。

O² 主配置只使用论文基准语义 `periodic_pi` angle loss，不再夹带
`square_aware_soft`。后者仍是框架中 direct-angle 版本可使用的本地稳健性扩展；
若以后研究 square ambiguity，必须放入独立 idea/config，不能修改 O² 复现基线。

## 4. Chamfer：论文与公开代码确实不一致

O²-DFINE 论文 Eq.10 写的是 squared corner distance；公开 O²-RTDETR
`ChamferCost` 使用普通 Euclidean norm。二者量纲和相对权重不同，不能用一个
模糊布尔值掩盖。

框架因此提供有名字的两种模式：

```text
paper_squared  -> O²-DFINE 论文公式（主 O² 配置）
released_l2    -> 公开 O²-RTDETR 源码（direct-angle 对照沿用）
```

matcher 只接受 `chamfer_distance` 具名模式，不再接受含义模糊的布尔别名。诊断
日志会写出所选模式和 source alignment。

## 5. KLD、matching 和 loss

当前主配置明确采用公开 O² 配置的 KLD 语义：

```text
sqrt=False, fun=log1p, tau=1
```

测试对全部公开 knob 与 ai4rs 参考实现逐值比较。几何 matching 使用 D-FINE 的
decoder/pre-decoder/encoder assignment union 监督，而分类仍是一对一；这是
D-FINE 原训练语义，不是 O² 单独提出的机制。

主 O² matcher 权重为 class 2、KLD 2、paper-squared Chamfer 5，bbox/angle cost
置零；criterion 仍保留必要的 bbox、angle、KLD 和 FGL 几何监督。所有选择均在
YAML 和诊断语义中显式记录，避免“看起来像 O²、实际混用了另一套 loss”。

## 6. OCD 复现边界

实现包含：

- box-frame vertex-coordinate noise；
- periodic angle noise；
- covariance/probability geometry noise；
- positive/negative DN groups；
- crowded-image 两种策略和完整选择/丢弃日志。

主 O² recipe 使用 `ocd_mode: box` 与 `released_dynamic`。这里
`num_denoising=200` 表示展开后的正负 query 总预算；公开配置中的 100 常表示
单侧/分组基数，不能按字段字面认定我们翻倍了 recipe。`released_dynamic` 优先
覆盖所有 GT，拥挤图像可超过请求预算；需要硬上限时可使用
`strict_budget_random`，但那是另一种明确记录的工程策略。

已知的论文零角退化等边界由测试原样记录，没有通过悄悄修改公式伪装一致。

## 7. Rotated cross-attention

当 reference 为 `(cx,cy,w,h,theta)` 时，deformable sampling offsets 先在局部框
坐标形成，再按 `theta` 旋转到图像坐标。这样斜长目标的采样方向跟随当前 query
几何，而不是始终水平/垂直。

它对 direct-angle 和 O² 两个 OBB 变体都生效，因为它是稳定 OBB decoder 能力，
不是 ADR 私有函数。测试验证 90° reference 会把水平 offset 精确旋成垂直，并且
诊断模式能保存 sampling location、rotated offsets 和每层证据。

## 8. 来源矩阵：哪些能称为复现，哪些不能

| 选择 | 来源 | 论文表述边界 |
|---|---|---|
| 六分布 ADR 与非均匀权重 | O²-DFINE 论文 | 可称 paper-derived reproduction |
| paper-squared Chamfer | O²-DFINE Eq.10 | 可称按论文公式实现 |
| `released_l2` Chamfer | 公开 O²-RTDETR | 只能称 released-source parity |
| KLD 与 OCD dynamic group | 公开 O² 配置/代码 | 可称公开实现对齐 |
| D-FINE union matching | D-FINE | 底座语义，不归属 O² |
| ordered-vertex 最小解码 / 闭式 KLD / raw-loss fail-loud | 本项目 | 复现边界与数值正确性 |
| square-aware soft angle loss | 本项目 direct-angle 可选语义 | 不属于 O² 主配置 |
| 通用 DOTA evaluator/日志 | 本项目框架 | 基础设施 |

因此最准确的模型名是“我们的 O²-DFINE reproduction”，而不是“官方
O²-DFINE”。若作者以后公开 DFINE 源码，必须重新做参数、target、loss reduction、
DN group、attention 和 inference 的逐项差分审计。

## 9. 冻结 checkpoint 的逐层全量验收（2026-08-24）

验收对象是 `logs/dfine_obb_o2/checkpoint0029.pth` 的 EMA 权重。数据链路明确为
CODrone `val` 的 2000 张**完整原图**，每张等比例 ResizePad 到 1024×1024；不是
tile 数据，也没有跨 tile 合并。checkpoint 使用严格加载，不允许 missing 或
unexpected key。

主验收对每一阶段使用同一套 flattened top-k、`score_threshold=.05` 和最多 500 个
检测，但关闭 overlap NMS：

| 阶段 | mAP50/75 DOTA-07 | AP50 | AP75 | mAP50:95 |
|---|---:|---:|---:|---:|
| prebox | 0.2043 | 0.2917 | 0.1169 | 0.1184 |
| layer 0 | 0.2057 | 0.2945 | 0.1168 | 0.1190 |
| layer 1 | 0.2175 | 0.3054 | 0.1297 | 0.1309 |
| layer 2 | 0.2288 | 0.3195 | 0.1382 | 0.1359 |
| layer 3 | **0.2324** | **0.3259** | **0.1390** | **0.1395** |

这给出了此前日志缺少的直接证据：ADR 不只改变匹配框，完整检测 AP 也从 layer 0
到最终层提高 0.0268，并且四个输出层的主指标严格递增。最后一层只增加 0.0036，
说明收益已明显衰减。

同一个最终输出加历史 rotated NMS 后为 0.2315，与训练日志的 0.2314 复现一致，
但低于无 NMS 的 0.2324。NMS 使最终输出从 704336 个降到 572326 个，并未改善
主指标；它是 full-image 配置继承来的历史后处理，不是 ResizePad 或 O² 所必需。
因此普通 O² 配置现在显式 `apply_nms: False`，tile 配置仍显式为 True。

89449 个最终 Hungarian match 的平均 rIoU 为：

```text
prebox 0.5548 -> layer0 0.5552 -> layer1 0.5670
       -> layer2 0.5689 -> layer3 0.5694
```

但对象级过程不是单调优化：58.66% 改善，35.26% 退化，只有 28.01% 在四层中
连续不下降。按 GT 事后给每个已匹配 query 选择五个阶段中最好的框，诊断上限为
0.2415，比固定最终层高 0.0091。这不是可报告方法结果，只说明 O² 仍有明显的
过修正/阶段选择空间。

自动选择的本质可视化（按 layer0→layer3 rIoU 极值选取，而非人工挑图）：

- [改善案例](../../../logs/dfine_obb_o2/acceptance/visualizations/improvement.png)：
  layer0 0.284 → layer3 0.930；
- [退化案例](../../../logs/dfine_obb_o2/acceptance/visualizations/degradation.png)：
  layer0 0.903 → layer3 0.310，主要崩塌发生在 layer1→layer2。

完整的逐层、per-class、NMS、oracle、source hash 和 gate 结果见
[`acceptance.json`](../../../logs/dfine_obb_o2/acceptance/acceptance.json)。可重复入口：

```bash
/icislab/volume1/liuxiaolong/anaconda3/envs/wyq-deim/bin/python -u \
  tools/research/o2/run_acceptance.py \
  --config configs/dfine/dfine_obb_o2.yml \
  --checkpoint logs/dfine_obb_o2/checkpoint0029.pth \
  --device cuda:1 --batch-size 4 --workers 0
```

验收状态为 **PASS**，范围是“当前 paper-derived O² 解码器确实训练、逐层精炼并在
完整验证集提高 AP”。它不改变第 8 节的来源边界，也不等价于官方未公开源码的逐行
parity。

## 10. 唯一入口和回归门槛

稳定配置：

```text
configs/dfine/dfine_obb_o2.yml
configs/dfine/dfine_obb_o2_tile.yml
```

核心测试：

- `test/research/o2/test_adr_geometry.py`：encode/decode、近轴、平移、square、
  chart seam、非法六值和有限梯度；
- `test/research/o2/test_dfine.py`：六分布 decoder、OCD、Chamfer、rotated
  attention、instability 和配置语义；
- `test/research/o2/test_training_parity.py`：KLD/Chamfer 来源差异、union
  matching、square extension、crowded DN；
- `test/framework/test_integration_contract.py`：canonical import、通用 evaluator
  注册和唯一 `refinement_mode` 契约。

任何后续 idea 都不得修改 O² 默认语义来获得自己的提升；应新增互斥
`refinement_mode` 或独立研究配置，并保持上述测试通过。
