# O²-DFINE 复现逻辑审计

## 结论与来源边界

`refinement_mode: o2_adr` 是 paper-derived O²-DFINE 复现，当前验收状态为
**PASS**（验收范围：本项目声明的 paper-derived O²-DFINE 完整原图复现契约）。
本地公开代码只包含 O²-RTDETR，
没有完整 O²-DFINE/ADR decoder；因此无法诚实地称作作者源码逐行复刻。实现选择按
以下优先级确定：

1. O²-DFINE 论文明确给出的结构、公式和超参数；
2. 公开 O²-RTDETR 的可执行训练语义；
3. O² 所继承的 D-FINE 原始语义；
4. 仅在前三者无法定义合法 OBB 时，采用已有 midpoint-offset 几何的确定性补全。

单元测试通过只说明局部契约成立。只有全新训练的 checkpoint 同时通过 loss/梯度、
完整验证集指标、逐层精炼和像素坐标几何验收后，状态才允许改为 **PASS**。

## UAV-ROD 从零训练证据（2026-08-25）

固定配置 `configs/experiments/uav_rod/dfine_obb_o2.yml` 以 seed 42 完成 72 轮训练；
训练实际构建 19,539,759 个总参数，其中 19,539,757 个可训练参数。按主指标保存的
`best_stg1.pth` 来自 epoch 67，最后一轮在更密集的 mAP50–95 上达到全程最高值。

- loss 从 35.351 降到 12.852；bbox、angle、KLD 分别从
  0.469/1.359/0.740 降到 0.015/0.110/0.010；
- 最后一轮完整验证集（427 图、11,461 个 GT）最终层 DOTA-07
  mean(AP50, AP75) 为 0.90846，diagnostic mAP50–95 为 0.89656；
- mAP50–95 按 `pre-box -> decoder 0 -> 1 -> 2 -> 3` 依次为
  `0.81423 -> 0.82474 -> 0.88243 -> 0.89217 -> 0.89656`；最终层相对
  pre-box 提升 0.08233，证明 ADR decoder 并非旁路或空转；
- 72 个 epoch 中有 65 个 epoch 的完整验证集 mAP50–95 逐层严格递增；详细记录的
  388 个有效 matched query 中，79.64% 的 pre-to-final rIoU 改善，19.85% 退化，
  均值从 0.86993 提高到 0.89925；单个目标不要求逐层单调；
- 206 个结构化训练采样点中，pre-box、ADR refinement 和 LQE 三类 head 均始终
  获得非零梯度。全程 10,251 step 只有 8 次 AMP 自动跳步，最低 scale 为 2048；
  除初始 scale 校准采样点外，其余 205 个采样点无非有限梯度；
- 完整原图验证关闭 NMS，19,056 个阈值后输出全部标记为 `kept`；当前结果不依赖
  rotated NMS。
- 当前配置的来源对齐契约独立检查为 PASS；`best_stg1.pth` 使用 epoch 67 的 EMA，
  对当前模型 strict load 为零 missing、零 unexpected。

训练日志证明了收敛、各 head 可训练、逐层输出有效且最终层带来完整验证集增益。

## 冻结 checkpoint 验收（2026-08-26）

`best_stg1.pth` 的独立完整验证集验收状态为 **PASS**。验收报告固定了配置、checkpoint
和关键源码 SHA-256；使用 epoch 67 的 EMA 权重、427 张完整图、1024 resize-pad、
无切片和无正式 NMS。全部必需 gate 均为 true：

- 来源对齐训练配置、pre-box 辅助监督、19.54M 模型规模和 strict load 通过；
- 固定 initial ADR anchor、上一层输出驱动下一层 query/reference、六值重建每层 OBB、
  LQE 加到分类 logits 四项真实 forward 误差均为 0（容差 `1e-6`）；
- model-space 与原图像素 rIoU 最大差为 `3.99e-6`（容差 `1e-3`）；
- 完整验证集 mAP50–95 按
  `pre-box -> layer 0 -> 1 -> 2 -> 3` 为
  `0.81294 -> 0.82285 -> 0.88051 -> 0.89044 -> 0.89632`，最终层相对
  pre-box 提升 `0.08337`；
- 全部 11,461 个 Hungarian matched object 的平均 rIoU 为
  `0.88897 -> 0.89473 -> 0.92317 -> 0.92693 -> 0.92805`；从 layer 0 到
  layer 3，82.82% 改善、17.01% 退化。

NMS 对照不是必需 gate。阈值 0.1 的 rotated NMS 删除 6,734/19,379（34.75%）个输出，
使粗粒度 DOTA-07 mean(AP50, AP75) 仅提高 `0.00021`，但使 mAP50–95 降低
`0.00227`、AP75 降低 `0.00752`。因此完整原图正式路径继续保持 NMS-free；该对照
不能解释为“模型需要 NMS 才成立”。

验收也保留了两个真实限制，不能因 PASS 而删除：

1. 只有 31.46% 的 matched object 从 layer 0 到 layer 3 逐层不退化；oracle 逐对象
   选层还能提高 `0.01045` mAP50–95，说明固定最后一层存在过精炼空间；
2. 六个独立期望在等对角闭合前的 raw orthogonality error 从 layer 0 均值 0.0253
   增至 layer 3 的 0.2154，且最终层 8.67% 不低于 0.5。确定性闭合保证最终输出始终
   是严格矩形且六值重建误差为 0，但论文没有公开 O²-DFINE 解码源码，因此不能把
   本项目的等对角补全宣称为作者逐行实现。这是已披露的来源边界，也是后续研究点。

机器可读结论位于
`logs/uav_rod/dfine_obb_o2/acceptance/acceptance.json`。在不改变上述源码、配置和
checkpoint 哈希的前提下，O² 基线不再需要重复训练或补做验收。

稳定入口只有：

```text
configs/dfine/dfine_obb_o2.yml
configs/dfine/dfine_obb_o2_tile.yml
```

## ADR：四边界与两偏移共同精炼 OBB

第一层 decoder 同时产生：

1. traditional OBB head 给出的初始 `(cx,cy,w,h,theta)`；
2. O²-DFINE head 给出的六组初始分布。

初始 OBB 转换为外接 HBox。前四组分布严格继承 D-FINE，对固定初始中心到
left/top/right/bottom 的距离进行累计 logits 精炼；后两组分布分别精炼：

- `epsilon`：外接框 top-right 到 OBB top vertex 的水平距离；
- `eta`：外接框 bottom-right 到 OBB right vertex 的竖直距离。

每层使用同一个初始 OBB 和初始外接尺寸作为锚：

```text
initial OBB
  -> initial external HBox + epsilon/eta
  -> cumulative six-distribution logits
  -> four refined HBox boundaries + two refined vertex offsets
  -> top/right/bottom/left vertices
  -> exact rectangle
  -> next-layer 5D rotated reference
```

六个量不是六个彼此独立的几何自由度，也不是“水平框之外再预测一个完整旋转框”。
前四个量与后两个量共同描述一个 OBB。合法矩形满足

\[
\epsilon(W_r-\epsilon)=\eta(H_r-\eta).
\]

六个独立分布的训练中间态不必天然满足这条关系。标准 midpoint/gliding-vertex
解码（MMRotate `MidpointOffsetCoder` 的既有规则）把 top/right 两条中心对角方向
缩放到二者较大的共同半径；中心对称且等对角线的
四边形必为矩形。该步骤只定义任意中间六值怎样落到合法 OBB，不增加预测量。
实现为纯 PyTorch，不调用 OpenCV。

## 四个 O² 机制

| 机制 | 当前实现 | 来源 |
|---|---|---|
| ADR | 六个非均匀分布、逐层累计 logits、FGL 监督六分量 | O²-DFINE 论文 |
| CDC | KLD + 四顶点 Chamfer Hungarian cost | 论文与公开 O² 代码 |
| OCD | box/angle/geometric/probability 原语，主配置为 box noise | 论文与公开 O² 代码 |
| Rotated cross-attention | 局部采样偏移按 OBB 方向旋转 | O² 论文描述 |

criterion 记录 decoder 层间 query-index instability。它是验证 OCD 动机的诊断量，
不是第五个模型模块。

ADR 图中的 `weighted distribution` 是固定解析码本的逐点乘积 `A(n)P(n)`，其求和
才是几何 residual。LQE 是另一条支路：它把六个分布的 top-k 概率统计送入 MLP，
得到 class logit correction。诊断日志分别记录二者，不能用 LQE 的存在解释
`A(n)P(n)`。

## CDC 与 KLD square problem

KLD 把 OBB 映射为高斯协方差。精确正方形的协方差与角度无关，因此 KLD 无法
区分同中心、同尺寸但几何朝向不同的正方形。CDC 直接比较四个角点集合：

- 相差 45° 的两个正方形角点集合不同，Chamfer 大于零；
- 相差 90° 的精确正方形本来就是同一几何，Chamfer 为零。

Chamfer 使用作者公开 O² 检测器中的可执行定义：四角点集合之间的双向最近邻
Euclidean distance 取均值。训练代码只提供这一种定义。

主 matcher 权重为 focal 2、KLD 2、Chamfer 5，bbox/angle matching cost 为零。
criterion 的 KLD 使用公开配置的 `sqrt=False, fun=log1p, tau=1`。

分类采用 D-FINE 与公开 O² 都使用的 Varifocal 语义（`alpha=0.75`）。VFL 与 FGL
的质量权重是 aligned XYWH IoU：只把前四个 `(cx,cy,w,h)` 参数解释为水平框，
不把角度混入质量权重。旋转质量由五参数 L1、KLD、正式 rIoU 指标和诊断分别负责。
公开 O² 对此明确选择 `hbox_iou`，D-FINE 的 FGL 也使用同一规则。

五参数 L1 对 `(cx,cy,w,h,theta/pi)` 直接计算归一化参数差，角度项权重为 5。
这保留公开 O² 的训练语义，不在复现中暗中改成周期或正方形感知 loss。

## OCD：为什么选择 box noise

主配置 `ocd_mode: box` 对齐公开 `only_xyxy` 路径：保持数值 theta 不变，
把 `(cx,cy,w,h)` 转为 `xyxy`，分别扰动四个坐标，再代数转换回
`(cx,cy,w,h)`。

论文消融中 box noise 最好。angle noise 只改变 theta，变化范围较窄；geometric
noise 同时改变 box 与 angle，容易造成过强 rIoU 下降；probability noise 主要
改变协方差，不能有效移动中心。

`num_denoising=100` 是正负展开前的动态分组基数，实际名义 query 数约为 200。
`released_dynamic` 在单图 GT 超过 100 时保留至少一组正负 query，因此实际数量
可以超过 200；日志记录名义数量、实际数量和 crowded policy。

## Rotated cross-attention

当 reference 是 `(cx,cy,w,h,theta)` 时，每个 D-FINE 非均匀 sampling offset
先在局部坐标按 `(w,h)` 缩放，再将完整二维向量按 theta 旋转到图像坐标，最后
加到中心位置。测试覆盖任意二维向量和各向异性 reference，不只覆盖 90° 特例。

公开 O²-RTDETR 实现将旋转矩阵作用于 `(w,h)` scale 后再与 offset 逐坐标相乘，
与论文“sampling points rotate”的描述并不等价。未公开的 O²-DFINE 无法逐行
核对；当前采用论文所描述的向量旋转，并在来源记录中明确披露该差异。

## ADR 是否存在 square problem

\[
(w,h,\theta)\equiv(h,w,\theta+\pi/2)
\]

不会产生两套 ADR target。ADR 的 top/right 是图像坐标中的全局极值顶点，不是
局部宽边/高边的顶点编号；等价参数化具有同一个角点集合，因此编码得到相同的
`epsilon/eta`。轴对齐 tie 使用固定的 top-right/bottom-right 规则，此时
`epsilon=eta=0`。

需要保留三个真实限制：

1. 框连续穿过图像坐标轴时，top/right 极值顶点可能换支，offset chart 存在 seam；
2. 五参数 raw L1 在 `0/pi` 边界可能远大于真实周期角误差；
3. 五参数 raw L1 对精确正方形的 90° 等价参数化仍可能产生非零损失。

日志同时记录 raw-L1 角误差、周期几何角误差、offset fraction、chart seam distance、
codebook overflow 和等对角闭合前的 raw orthogonality error，使这些问题可以直接
从实验日志分析。

## 当前实验与诊断契约

当前唯一训练预算采用论文 O²-DFINE-M 设置：HGNetv2-B2、300 matching queries、
四层 decoder、总 batch size 8、72 epochs、base/backbone LR 为 `5e-5/5e-6`、
weight decay `1e-4`、gradient clip `0.1`。HybridEncoder 使用 M 配置的
`expansion=1.0, depth_mult=0.67`；UAV-ROD recipe 实际构建为 19,539,759 个总参数，
其中 19,539,757 个可训练参数，落在论文报告的约 19M 规模。UAV-ROD 与后续 CODrone
使用同一模型和优化预算，
数据集配置只注入数据路径、类别数和增强。

几何增强顺序与公开 O² pipeline 对齐为：保持比例缩放、随机翻转、随机旋转、
旋转后实例清理、右下补边、张量化。旋转发生在补边之前，因此非正方形图像围绕
真实缩放图像中心旋转；主 recipe 不加入公开流程中没有的光度增强、Mosaic 或 MixUp。

完整原图推理不执行 rotated NMS；tile 配置在预测平移回源图后执行类别感知
rotated NMS。

训练后使用
`tools/research/o2/run_acceptance.py` 在同一完整验证集上报告 prebox、每个
decoder layer、最终输出、NMS 对照、逐类 AP、matched rIoU、过修正案例和
raw ADR consistency。所有报告的 rIoU、中心误差和尺寸误差都在原图像素/弧度
坐标计算；工具还会独立计算同一几何的 model-space rIoU，二者超出容差立即失败。
rIoU 由同一套 float64 凸四边形交集实现计算，并在生成角点前完成精度提升；正式
评估、诊断和 rotated NMS 因而共享完全相同的几何语义。

验收还从真实 forward tensor 检查四条结构事实：所有层使用同一个 initial ADR
anchor；上一层 OBB 确实成为下一层 query/attention reference；四边界加两偏移能
重建每层输出 OBB；LQE correction 确实加入分类 logit。这些是“精炼真的发生”的
最低条件，不能用最终 AP 偶然较高代替。

`configs/experiments/uav_rod/dfine_obb_o2.yml` 是低成本的几何/精炼验收 run：每个
epoch 记录完整验证集 pre-box 与各 decoder stage 的 AP，每五个 epoch 记录固定前
24 张图的 object-level 分布与框。它只有 car 一类，因此适合 ADR、逐层 reference
传递和 LQE 诊断，但不能单独验证 CDC 的多类别匹配收益或 label-noise 的类别扰动收益。
UAV-ROD 的 labelled test split 在这里仅作机制验收集，不能把反复查看后的结果当作
无偏论文测试结果。CODrone 的多类别、密集小目标和更高分辨率阶段必须重新验证 CDC、
固定 query 容量、resize 信息损失和类间重复预测。

论文图由 `tools/research/o2/visualize_mechanisms.py` 从日志自动生成。默认例子按预先
声明的数值规则选择（跨 epoch 最大提升、最新 pre-to-final 最大提升、最大过修正/最小
提升、最终 IoU 中位样本），不看图像外观；选择规则、对象 ID、query ID 与数值均写入
`evidence_index.json`。

## 回归门槛

- `test/research/o2/test_adr_geometry.py`：encode/decode、轴边界、平移、
  square、非法六值和完整 KLD backward；
- `test/research/o2/test_dfine.py`：累计六分布、OCD、CDC、旋转采样和配置；
- `test/research/o2/test_training_parity.py`：来源语义、square matching、
  aligned-XYWH 质量、raw L1 局限、union matching 和 crowded DN；
- `test/research/o2/test_visualization.py`：固定对象选择、逐 epoch/逐层框、
  ADR unweighted/weighted 分布与全验证集 stage 曲线；
- `test/framework/model/test_model_pipeline.py`：训练、推理和后处理；
- `test/framework/model/test_full_smoke.py`：真实 1024 CUDA autocast
  forward、loss、backward 和 inference。

任何后续方法必须使用独立 `refinement_mode` 或研究配置，不得修改 O² 默认语义。
